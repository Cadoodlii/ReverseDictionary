import json
import random
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalRD(Dataset):
  """Multimodal reverse dictionary dataset."""

  def __init__(
    self,
    json_path: str,
    glove_emb: Dict[str, np.ndarray],
    transform: Optional[Callable] = None,
  ):
    """
    Args:
      json_path: path to the pseudo-RDMIF JSON file.
      glove_emb: dict word -> np.ndarray from load_glove_embeddings().
      transform: torchvision transform for images. If None, a basic
        resize/crop/ToTensor pipeline is used.
    """
    with open(json_path, "r", encoding="utf-8") as f:
      raw_items = json.load(f)

    self.glove = glove_emb

    if transform is not None:
      self.transform = transform
    else:
      # Use ImageNet normalization for ResNet
      self.transform = transforms.Compose(
        [
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
          ),
        ]
      )

    if len(self.glove) > 0:
      self.emb_dim = len(next(iter(self.glove.values())))

    # Filter items that actually have images
    items = []
    for it in raw_items:
      img_dir = Path(it.get("image_dir", ""))
      files = []
      if img_dir.exists() and img_dir.is_dir():
        for path in img_dir.iterdir():
          if path.is_file():
            files.append(str(path))

      if not files:
        continue  # drop items with no images

      it = dict(it)  # avoid mutating the original list
      it["files"] = files
      it["word"] = it["word"].lower()
      items.append(it)

    self.items = items

    # Build vocab & mappings once
    vocab = sorted({it["word"] for it in self.items})
    self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
    self.idx2word: Dict[int, str] = {i: w for w, i in self.word2idx.items()}

    for it in self.items:
      it["word_idx"] = self.word2idx[it["word"]]

    # GloVe matrix aligned with this vocab
    self.glove_matrix = np.zeros((len(vocab), self.emb_dim), dtype=np.float32)
    for w, idx in self.word2idx.items():
      self.glove_matrix[idx] = self._label_to_glove_vec(w)

  def _label_to_glove_vec(self, label: str) -> np.ndarray:
    """Get a GloVe vector for a label like 'great_white_shark'.

    Strategy:
      1) if full label exists in GloVe, use it
      2) else split on '_' and average token embeddings
      3) if nothing found, return OOV vector (random)
    """
    label = label.lower()

    # Full phrase in GloVe
    if label in self.glove:
      return self.glove[label]

    # Split on "_" and average what we find
    tokens = label.split("_")
    vecs = [self.glove[t] for t in tokens if t in self.glove]

    if vecs:
      vecs = np.stack(vecs, axis=0)  # [n_tokens, emb_dim]
      return vecs.mean(axis=0)
    # OOV fallback
    else:
      return np.random.normal(loc=0.0, scale=0.01, size=(self.emb_dim,)).astype(
        np.float32
      )

  def __len__(self):
    return len(self.items)

  def __getitem__(self, index: int):
    item = self.items[index]
    files = item["files"]

    file_path = random.choice(files)  # Pick a random image
    img = Image.open(file_path).convert("RGB")  # Load image from file_path

    # Transform image
    if self.transform:
      img = self.transform(img)

    return {
      "text": item["definition"],
      "image": img,
      "word": item["word"],
      "word_idx": item["word_idx"],
      "hypernym": item["hypernym"],
    }
