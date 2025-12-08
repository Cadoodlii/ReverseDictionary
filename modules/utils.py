from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.models import FusionModel


def set_seed(seed: int) -> None:
  """Set random seeds for reproducibility."""
  import random

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_glove_embeddings(glove_path: str) -> Dict[str, np.ndarray]:
  """Load GloVe vectors from a text file into a dict: word -> numpy array.

  The file is expected to have the usual "word dim1 dim2 ... dimN" format.
  All words are lowercased.
  """
  emb: Dict[str, np.ndarray] = {}

  with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
      parts = line.strip().split()
      if not parts:
        continue
      word, *vec = parts
      emb[word.lower()] = np.asarray(vec, dtype=np.float32)

  return emb


def contrastive_loss(
  a: torch.Tensor,
  b: torch.Tensor,
  temperature: float = 0.07,
) -> torch.Tensor:
  """Symmetric InfoNCE between two modalities.

  Args:
      a, b:
          Tensors of shape [B, D]. They do not need to be normalized;
          they are L2-normalized inside this function for stability.
      temperature:
          Softmax temperature. Smaller values sharpen the distribution.

  Returns:
      A scalar tensor containing the average of the two directions:
      a->b and b->a.
  """
  if a.ndim != 2 or b.ndim != 2:
    raise ValueError(f"Expected [B, D] tensors, got a:{a.shape}, b:{b.shape}")

  a = F.normalize(a, dim=-1)
  b = F.normalize(b, dim=-1)

  # Similarity matrix: each a_i compared with all b_j
  logits = a @ b.t() / temperature  # [B, B]
  labels = torch.arange(a.size(0), device=a.device)

  loss_ab = F.cross_entropy(logits, labels)  # a -> b
  loss_ba = F.cross_entropy(logits.t(), labels)  # b -> a
  return 0.5 * (loss_ab + loss_ba)


def contrastive_loss_one_sided(
  a: torch.Tensor,
  b: torch.Tensor,
  temperature: float = 0.07,
) -> torch.Tensor:
  """One-sided InfoNCE: a -> b."""
  if a.ndim != 2 or b.ndim != 2:
    raise ValueError(f"Expected [B, D] tensors, got a:{a.shape}, b:{b.shape}")

  a = F.normalize(a, dim=-1)
  b = F.normalize(b, dim=-1)

  logits = a @ b.t() / temperature  # [B, B]
  labels = torch.arange(a.size(0), device=a.device)
  loss = F.cross_entropy(logits, labels)
  return loss


@torch.no_grad()
def _evaluate_reverse_dict(
  model,
  dataloader: DataLoader,
  device: torch.device,
  use_image: bool = False,
):
  """
  Generic reverse-dictionary evaluation.

  use_image = False  -> text-only (definitions)
  use_image = True   -> fuse text and image (z = norm(z_text + z_img))
  """
  model.eval()

  # Get all word embeddings from the model vocab
  vocab_size = model.word_emb.num_embeddings
  all_indices = torch.arange(vocab_size, device=device)  # [V]
  all_word_emb = model.embed_words(all_indices)  # [V, D] (normalized)

  total = 0
  sum_cos = 0.0
  recall1 = 0
  recall5 = 0
  recall10 = 0
  all_ranks = []

  # ======================== INFERENCE ======================== #
  for batch in dataloader:
    input_ids = batch["input_ids"].to(device)  # [B, L]
    attention_mask = batch["attention_mask"].to(device)
    word_idx = batch["word_idx"].to(device)  # [B]
    B = word_idx.size(0)

    # Encode text
    z_text = model.encode_text(input_ids, attention_mask)  # [B, D]

    if isinstance(model, FusionModel):
      if use_image:
        images = batch["images"].to(device)
        z_img = model.encode_image(images)
        z = model.fuse(z_text, z_img)
      else:
        z = model.fuse(z_text, torch.zeros_like(z_text))  # Send in zeros for image
    else:
      if use_image:
        images = batch["images"].to(device)
        z_img = model.encode_image(images)
        z = F.normalize((z_text + z_img) / 2, dim=-1)
      else:
        z = z_text

    # Similarity to all words: [B, V]
    sims = z @ all_word_emb.t()

    # ----- Recall@K -----
    K = 10
    topk_idx = sims.topk(k=K, dim=-1).indices  # [B, K]

    # R@1
    recall1 += (topk_idx[:, 0] == word_idx).sum().item()

    # R@5
    in_top5 = (topk_idx[:, :5] == word_idx.unsqueeze(1)).any(dim=1)
    recall5 += in_top5.sum().item()

    # R@10
    in_top10 = (topk_idx[:, :10] == word_idx.unsqueeze(1)).any(dim=1)
    recall10 += in_top10.sum().item()

    # ----- Rank (for median rank) -----
    ranks = torch.argsort(sims, dim=-1, descending=True)  # [B, V]
    mask = ranks == word_idx.unsqueeze(1)  # [B, V]
    positions = torch.nonzero(mask, as_tuple=False)[:, 1]  # [B], 0-based
    batch_ranks = (positions + 1).tolist()  # convert to 1-based rank
    all_ranks.extend(batch_ranks)

    # Cosine similarity with *correct* word embedding
    true_word_emb = all_word_emb[word_idx]  # [B, D]
    cos_sim = (z * true_word_emb).sum(dim=-1)  # [B]
    sum_cos += cos_sim.sum().item()

    total += B

  r_at_1 = recall1 / total
  r_at_5 = recall5 / total
  r_at_10 = recall10 / total
  median_rank = float(np.median(all_ranks))
  avg_cosine = sum_cos / total

  return {
    "R@1": r_at_1,
    "R@5": r_at_5,
    "R@10": r_at_10,
    "median_rank": median_rank,
    "avg_cosine": avg_cosine,
  }


@torch.no_grad()
def evaluate_text_only(
  model,
  dataloader: DataLoader,
  device: torch.device,
):
  """Reverse-dictionary evaluation using *only definitions*."""
  return _evaluate_reverse_dict(model, dataloader, device, use_image=False)


@torch.no_grad()
def evaluate_text_image(
  model,
  dataloader: DataLoader,
  device: torch.device,
):
  """Reverse-dictionary evaluation using *definitions + images*."""
  return _evaluate_reverse_dict(model, dataloader, device, use_image=True)


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
  """Pretty-print metrics with a given prefix."""
  return (
    f"[{prefix}] "
    f"R@1={metrics['R@1']:.4f}, "
    f"R@5={metrics['R@5']:.4f}, "
    f"R@10={metrics['R@10']:.4f}, "
    f"med_rank={metrics['median_rank']:.2f}, "
    f"cos={metrics['avg_cosine']:.4f}"
  )


def maybe_save_checkpoint(
  cfg,
  model,
  optimizer: torch.optim.Optimizer,
  scheduler,
  epoch: int,
  text_metrics,
  mm_metrics,
  best_r10: float,
  best_text: bool = True,
) -> float:
  """
  Save checkpoint if R@10 (text-only) improves and saving is enabled.
  Returns the updated best R@10.
  """
  if best_text:
    current_r10 = text_metrics["R@10"]
  else:
    current_r10 = mm_metrics["R@10"]

  if not cfg.save_best:
    return max(current_r10, best_r10)

  if current_r10 > best_r10:
    best_r10 = current_r10
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
      "text_metrics": text_metrics,
      "mm_metrics": mm_metrics,
      "best_R@10": best_r10,
    }
    torch.save(state, cfg.ckpt_path)
    print(f"  >> Saved new best checkpoint to {cfg.ckpt_path} (R@10={best_r10:.4f})")
  return best_r10


def load_checkpoint(
  cfg,
  model: torch.nn.Module,
  optimizer: Optional[torch.optim.Optimizer] = None,
  scheduler: Optional[Any] = None,
  map_location: str | torch.device = "cpu",
) -> Tuple[int, float, Dict, Dict]:
  """
  Load checkpoint from cfg.ckpt_path into model (and optionally optimizer/scheduler).

  Returns:
      start_epoch: int
          Epoch to resume from (saved_epoch + 1).
      best_r10: float
          Best R@10 stored in the checkpoint.
      text_metrics: dict
      mm_metrics: dict
  """
  ckpt_path = Path(cfg.ckpt_path)
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  print(f"Loading checkpoint from {ckpt_path}")
  state = torch.load(ckpt_path, map_location=map_location)

  # Restore model
  model.load_state_dict(state["model_state_dict"])

  # Restore optimizer if provided
  if optimizer is not None and state.get("optimizer_state_dict") is not None:
    optimizer.load_state_dict(state["optimizer_state_dict"])

  # Restore scheduler if provided
  if scheduler is not None and state.get("scheduler_state_dict") is not None:
    # Some schedulers may store None, so guard it
    if state["scheduler_state_dict"] is not None:
      scheduler.load_state_dict(state["scheduler_state_dict"])

  epoch = state.get("epoch", 0)
  best_r10 = state.get("best_R@10", 0.0)
  text_metrics = state.get("text_metrics", {})
  mm_metrics = state.get("mm_metrics", {})

  # Typically you resume from the *next* epoch
  start_epoch = epoch + 1

  print(f"  >> Resumed from epoch {epoch}, best R@10={best_r10:.4f}")

  return start_epoch, best_r10, text_metrics, mm_metrics
