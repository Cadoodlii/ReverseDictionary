from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
  ResNet18_Weights,
  ResNet50_Weights,
  resnet18,
  resnet50,
)
from transformers import AutoModel

_BACKBONE_TYPE = Literal["resnet18", "resnet50"]


def _build_image_backbone(
  backbone: _BACKBONE_TYPE,
  pretrained: bool = True,
) -> tuple[nn.Module, int]:
  """Construct an ImageNet backbone and return (encoder, feature_dim)."""
  if backbone == "resnet18":
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    feature_dim = 512
  elif backbone == "resnet50":
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    feature_dim = 2048
  else:
    raise ValueError(f"Unsupported backbone: {backbone}")

  # Remove the classification head; keep only the convolutional trunk.
  model.fc = nn.Identity()
  return model, feature_dim


class TextModel(nn.Module):
  """Text only model."""

  def __init__(
    self,
    glove_matrix: torch.Tensor,
    text_model_name: str = "distilbert-base-uncased",
    freeze_text: bool = False,
    freeze_word_embeddings: bool = True,
  ) -> None:
    super().__init__()

    glove_matrix = glove_matrix.float()
    num_words, emb_dim = glove_matrix.shape
    self.emb_dim = emb_dim
    self.vocab_size = num_words

    # ----- Text encoder -----
    self.text_encoder = AutoModel.from_pretrained(text_model_name)
    text_hidden_dim = self.text_encoder.config.hidden_size

    # ----- Projection heads into the shared embedding space -----
    self.text_proj = nn.Sequential(
      nn.Linear(text_hidden_dim, text_hidden_dim),
      nn.ReLU(),
      nn.Linear(text_hidden_dim, emb_dim),
    )

    # ----- Word embeddings: initialised from GloVe -----
    self.word_emb = nn.Embedding.from_pretrained(
      glove_matrix,
      freeze=freeze_word_embeddings,
    )

    # Optionally freeze encoders (projection heads stay trainable)
    if freeze_text:
      for p in self.text_encoder.parameters():
        p.requires_grad = False

  # ------------------------------------------------------------------ #
  # Encoders
  # ------------------------------------------------------------------ #

  def encode_text(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Encode a batch of tokenized definitions into L2-normalized vectors."""
    out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls = out.last_hidden_state[:, 0]  # [B, H]
    z = self.text_proj(cls)  # [B, D]
    return F.normalize(z, dim=-1)

  def embed_words(self, word_idx: torch.Tensor) -> torch.Tensor:
    """Lookup and L2-normalize word embeddings.

    `word_idx` can be a batch of indices [B] or all vocab indices [V].
    """
    w = self.word_emb(word_idx)  # [B, D] or [V, D]
    return F.normalize(w, dim=-1)

  # ------------------------------------------------------------------ #
  # Convenience forward
  # ------------------------------------------------------------------ #

  def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    word_idx: Optional[torch.Tensor] = None,
  ) -> dict:
    """Optional forward that returns any requested embeddings.

    This makes it easier to use with generic training loops if desired.
    """
    outputs: dict = {}
    if input_ids is not None:
      outputs["z_text"] = self.encode_text(input_ids, attention_mask)
    if word_idx is not None:
      outputs["z_word"] = self.embed_words(word_idx)
    return outputs


class FusionModel(nn.Module):
  """Image+text fusion model..

  - Encodes text with a Transformer (e.g. DistilBERT) -> [B, H_text]
    (via CLS pooling + projection).
  - Encodes image with ResNet backbone -> [B, H_img] -> projection.
  - Fusion options:

    (1) "gated_add" (default):
        gate = sigmoid(w)  # [D]
        fused = gate * z_text + (1 - gate) * z_img

    (2) "concat":
        fused = MLP([z_text ; z_img])  # [B, 2D] -> [B, D]

  - Returns L2-normalized fused embedding [B, emb_dim].
  """

  def __init__(
    self,
    glove_matrix: torch.Tensor,
    text_model_name: str = "distilbert-base-uncased",
    image_backbone: _BACKBONE_TYPE = "resnet50",
    freeze_text: bool = False,
    freeze_image: bool = False,
    freeze_word_embeddings: bool = True,
    fusion_type: str = "gated_add",
    p_drop_image: float = 0.3,
  ) -> None:
    super().__init__()

    glove_matrix = glove_matrix.float()
    num_words, emb_dim = glove_matrix.shape
    self.emb_dim = emb_dim
    self.vocab_size = num_words

    # Sanity-check fusion type
    if fusion_type not in {"gated_add", "concat"}:
      raise ValueError(
        f"fusion_type must be 'gated_add' or 'concat', got {fusion_type}"
      )
    self.fusion_type = fusion_type

    self.p_drop_image = p_drop_image

    # ----- Text encoder -----
    self.text_encoder = AutoModel.from_pretrained(text_model_name)
    text_hidden_dim = self.text_encoder.config.hidden_size  # e.g. 768

    # ----- Image encoder -----
    img_encoder, img_hidden_dim = _build_image_backbone(
      backbone=image_backbone,
      pretrained=True,
    )
    self.img_encoder = img_encoder

    # ----- Projection heads into shared embedding space -----
    self.text_proj = nn.Sequential(
      nn.Linear(text_hidden_dim, text_hidden_dim),
      nn.ReLU(),
      nn.Linear(text_hidden_dim, emb_dim),
    )

    self.img_proj = nn.Sequential(
      nn.Linear(img_hidden_dim, img_hidden_dim),
      nn.ReLU(),
      nn.Linear(img_hidden_dim, emb_dim),
    )

    # ----- Learnable fusion gate (for "gated_add" mode) -----
    self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    # ----- Fusion MLP (for "concat" mode) -----
    # Defined unconditionally; only used when fusion_type == "concat".
    self.fusion_proj = nn.Sequential(
      nn.Linear(2 * emb_dim, 2 * emb_dim),
      nn.ReLU(),
      nn.Linear(2 * emb_dim, emb_dim),
    )

    # ----- Word embeddings: initialised from GloVe -----
    self.word_emb = nn.Embedding.from_pretrained(
      glove_matrix,
      freeze=freeze_word_embeddings,
    )

    # Optional freezing
    if freeze_text:
      for p in self.text_encoder.parameters():
        p.requires_grad = False
    if freeze_image:
      for p in self.img_encoder.parameters():
        p.requires_grad = False

  # ------------------------------------------------------------------ #
  # Encoders
  # ------------------------------------------------------------------ #

  def encode_text(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Encode text into L2-normalized embedding [B, emb_dim]."""
    out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls = out.last_hidden_state[:, 0]  # [B, H_text]
    z = self.text_proj(cls)  # [B, D]
    return F.normalize(z, dim=-1)

  def encode_image(self, images: torch.Tensor) -> torch.Tensor:
    """Encode image into L2-normalized embedding [B, emb_dim]."""
    h = self.img_encoder(images)  # [B, H_img]
    z = self.img_proj(h)  # [B, D]
    return F.normalize(z, dim=-1)

  def embed_words(self, word_idx: torch.Tensor) -> torch.Tensor:
    """Lookup and L2-normalize word embeddings.

    `word_idx` can be a batch of indices [B] or all vocab indices [V].
    """
    w = self.word_emb(word_idx)  # [B, D] or [V, D]
    return F.normalize(w, dim=-1)

  # ------------------------------------------------------------------ #
  # Fusion
  # ------------------------------------------------------------------ #

  def fuse(
    self,
    z_text: torch.Tensor,
    z_img: torch.Tensor,
  ) -> torch.Tensor:
    """Fuse text + image embeddings."""
    # Modality dropout
    if self.training and self.p_drop_image > 0.0:
      B = z_img.size(0)
      device = z_img.device

      # Per-sample Bernoulli mask: True means "drop image"
      drop_mask = torch.rand(B, 1, device=device) < self.p_drop_image
      # Zero-out image embeddings where mask is True
      z_img = z_img.masked_fill(drop_mask, 0.0)

    if self.fusion_type == "gated_add":
      gate = torch.sigmoid(self.fusion_gate)  # [D]
      fused = gate * z_text + (1.0 - gate) * z_img  # [B, D]
    elif self.fusion_type == "concat":
      concat = torch.cat([z_text, z_img], dim=-1)  # [B, 2D]
      fused = self.fusion_proj(concat)  # [B, D]
    return F.normalize(fused, dim=-1)

  # ------------------------------------------------------------------ #
  # Forward
  # ------------------------------------------------------------------ #

  def forward(
    self,
    images: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    word_idx: Optional[torch.Tensor] = None,
  ) -> dict:
    """Optional forward that returns any requested embeddings.

    This makes it easier to use with generic training loops if desired.
    """
    outputs: dict = {}
    if input_ids is not None:
      outputs["z_text"] = self.encode_text(input_ids, attention_mask)
    if images is not None:
      outputs["z_image"] = self.encode_image(images)
    if input_ids is not None and images is not None:
      outputs["z_fused"] = self.fuse(outputs["z_text"], outputs["z_image"])
    if word_idx is not None:
      outputs["z_word"] = self.embed_words(word_idx)
    return outputs


class ContrastiveModel(nn.Module):
  """Joint text-image-word embedding model for contrastive training.

  Args:
      glove_matrix:
          Precomputed GloVe matrix of shape [V, D] aligned with dataset vocab.
      text_model_name:
          HuggingFace model name for the text encoder (e.g. "distilbert-base-uncased").
      image_backbone:
          Which ResNet backbone to use ("resnet18" or "resnet50").
      freeze_text:
          If True, freeze the text encoder weights.
      freeze_image:
          If True, freeze the image encoder weights.
      freeze_word_embeddings:
          If True (default), do not fine-tune the GloVe matrix.
  """

  def __init__(
    self,
    glove_matrix: torch.Tensor,
    text_model_name: str = "distilbert-base-uncased",
    image_backbone: _BACKBONE_TYPE = "resnet50",
    freeze_text: bool = False,
    freeze_image: bool = False,
    freeze_word_embeddings: bool = True,
  ) -> None:
    super().__init__()

    glove_matrix = glove_matrix.float()
    num_words, emb_dim = glove_matrix.shape
    self.emb_dim = emb_dim
    self.vocab_size = num_words

    # ----- Text encoder -----
    self.text_encoder = AutoModel.from_pretrained(text_model_name)
    text_hidden_dim = self.text_encoder.config.hidden_size

    # ----- Image encoder -----
    img_encoder, img_hidden_dim = _build_image_backbone(
      backbone=image_backbone,
      pretrained=True,
    )
    self.img_encoder = img_encoder

    # ----- Projection heads into the shared embedding space -----
    self.text_proj = nn.Sequential(
      nn.Linear(text_hidden_dim, text_hidden_dim),
      nn.ReLU(),
      nn.Linear(text_hidden_dim, emb_dim),
    )
    self.img_proj = nn.Sequential(
      nn.Linear(img_hidden_dim, img_hidden_dim),
      nn.ReLU(),
      nn.Linear(img_hidden_dim, emb_dim),
    )

    # ----- Word embeddings: initialised from GloVe -----
    self.word_emb = nn.Embedding.from_pretrained(
      glove_matrix,
      freeze=freeze_word_embeddings,
    )

    # Optionally freeze encoders (projection heads stay trainable)
    if freeze_text:
      for p in self.text_encoder.parameters():
        p.requires_grad = False
    if freeze_image:
      for p in self.img_encoder.parameters():
        p.requires_grad = False

  # ------------------------------------------------------------------ #
  # Encoders
  # ------------------------------------------------------------------ #

  def encode_text(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Encode a batch of tokenized definitions into L2-normalized vectors."""
    out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls = out.last_hidden_state[:, 0]  # [B, H]
    z = self.text_proj(cls)  # [B, D]
    return F.normalize(z, dim=-1)

  def encode_image(self, images: torch.Tensor) -> torch.Tensor:
    """Encode a batch of images into L2-normalized vectors."""
    h = self.img_encoder(images)  # [B, C]
    z = self.img_proj(h)  # [B, D]
    return F.normalize(z, dim=-1)

  def embed_words(self, word_idx: torch.Tensor) -> torch.Tensor:
    """Lookup and L2-normalize word embeddings.

    `word_idx` can be a batch of indices [B] or all vocab indices [V].
    """
    w = self.word_emb(word_idx)  # [B, D] or [V, D]
    return F.normalize(w, dim=-1)

  # ------------------------------------------------------------------ #
  # Convenience forward
  # ------------------------------------------------------------------ #

  def forward(
    self,
    images: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    word_idx: Optional[torch.Tensor] = None,
  ) -> dict:
    """Optional forward that returns any requested embeddings.

    This makes it easier to use with generic training loops if desired.
    """
    outputs: dict = {}
    if input_ids is not None:
      outputs["z_text"] = self.encode_text(input_ids, attention_mask)
    if images is not None:
      outputs["z_image"] = self.encode_image(images)
    if word_idx is not None:
      outputs["z_word"] = self.embed_words(word_idx)
    return outputs


class CLIPContrastiveModel(nn.Module):
  """Joint text-image-word embedding model using CLIP encoders.

  Args:
    glove_matrix:
      Precomputed GloVe matrix of shape [V, D] aligned with dataset vocab.
    clip_model_name:
      HuggingFace model name for CLIP (e.g. "openai/clip-vit-base-patch32").
    freeze_text:
      If True, freeze CLIP's text-side parameters.
    freeze_image:
      If True, freeze CLIP's vision-side parameters.
    freeze_word_embeddings:
      If True (default), do not fine-tune the GloVe matrix.

  Notes:
    - Text inputs should be tokenized with the matching CLIP tokenizer
      (e.g. CLIPTokenizer / AutoTokenizer.from_pretrained(clip_model_name)).
    - Image inputs should be preprocessed with the corresponding CLIP image
      processor (e.g. CLIPImageProcessor), and passed as `images` with
      shape [B, 3, H, W], dtype float32.
  """

  def __init__(
    self,
    glove_matrix: torch.Tensor,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    freeze_text: bool = False,
    freeze_image: bool = False,
    freeze_word_embeddings: bool = True,
  ) -> None:
    super().__init__()

    glove_matrix = glove_matrix.float()
    num_words, emb_dim = glove_matrix.shape
    self.emb_dim = emb_dim
    self.vocab_size = num_words

    # ----- CLIP encoders (text + vision) -----
    self.clip = AutoModel.from_pretrained(clip_model_name)
    clip_emb_dim = self.clip.config.projection_dim  # typically 512

    # ----- Projection heads into the shared embedding space (GloVe dim) -----
    self.text_proj = nn.Sequential(
      nn.Linear(clip_emb_dim, clip_emb_dim),
      nn.ReLU(),
      nn.Linear(clip_emb_dim, emb_dim),
    )
    self.img_proj = nn.Sequential(
      nn.Linear(clip_emb_dim, clip_emb_dim),
      nn.ReLU(),
      nn.Linear(clip_emb_dim, emb_dim),
    )

    # ----- Word embeddings: initialised from GloVe -----
    self.word_emb = nn.Embedding.from_pretrained(
      glove_matrix,
      freeze=freeze_word_embeddings,
    )

    # Optionally freeze CLIP encoders
    if freeze_text:
      for p in self.clip.text_model.parameters():
        p.requires_grad = False
      # text_projection is an nn.Parameter
      self.clip.text_projection.requires_grad = False

    if freeze_image:
      for p in self.clip.vision_model.parameters():
        p.requires_grad = False
      # visual_projection is an nn.Parameter
      self.clip.visual_projection.requires_grad = False

  # ------------------------------------------------------------------ #
  # Encoders
  # ------------------------------------------------------------------ #

  def encode_text(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Encode a batch of tokenized definitions into L2-normalized vectors."""
    # CLIP will handle positional embeddings, projections, etc.
    text_feats = self.clip.get_text_features(
      input_ids=input_ids,
      attention_mask=attention_mask,
    )  # [B, clip_emb_dim]

    z = self.text_proj(text_feats)  # [B, D]
    return F.normalize(z, dim=-1)

  def encode_image(self, images: torch.Tensor) -> torch.Tensor:
    """Encode a batch of images (CLIP-preprocessed) into L2-normalized vectors."""
    img_feats = self.clip.get_image_features(
      pixel_values=images,
    )  # [B, clip_emb_dim]

    z = self.img_proj(img_feats)  # [B, D]
    return F.normalize(z, dim=-1)

  def embed_words(self, word_idx: torch.Tensor) -> torch.Tensor:
    """Lookup and L2-normalize word embeddings.

    `word_idx` can be a batch of indices [B] or all vocab indices [V].
    """
    w = self.word_emb(word_idx)  # [B, D] or [V, D]
    return F.normalize(w, dim=-1)

  # ------------------------------------------------------------------ #
  # Convenience forward
  # ------------------------------------------------------------------ #

  def forward(
    self,
    images: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    word_idx: Optional[torch.Tensor] = None,
  ) -> dict:
    """Optional forward that returns any requested embeddings.

    This mirrors your original API to make swapping models easy.
    """
    outputs: dict = {}
    if input_ids is not None:
      outputs["z_text"] = self.encode_text(input_ids, attention_mask)
    if images is not None:
      outputs["z_image"] = self.encode_image(images)
    if word_idx is not None:
      outputs["z_word"] = self.embed_words(word_idx)
    return outputs
