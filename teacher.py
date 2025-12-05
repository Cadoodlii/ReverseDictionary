"""Teacher model for Reverse Dictionary (Multimodal RDMIF-style).

Architecture:
- Text Encoder: BERT (AutoModel) for definition encoding
- Image Encoder: ResNet-50 (pre-trained) for image encoding
- Fusion: Concatenate text + image -> Linear layer -> target embedding dim (300 for GloVe)

Loss: MSELoss between predicted vector and ground-truth GloVe vector.

Co-Learning: Modality Dropout
- During training, randomly zero out image_tensor (with probability modality_dropout_p)
- Forces model to rely on text when images are "missing"
- Encourages robustness and balanced multimodal learning

Utilities included:
- TeacherModel (torch.nn.Module)
- TeacherDataset to pair definitions + images -> glove vector
- train/evaluate helper functions and CLI example

Requirements:
  pip install torch transformers numpy tqdm torchvision pillow

Run example (after preparing `dataset.json` and GloVe file):
  python3 teacher.py --dataset dataset.json --glove /path/to/glove.6B.300d.txt --epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None

try:
    import torchvision.models as models
    from torchvision import transforms as tv_transforms
except Exception:
    models = None
    tv_transforms = None

try:
    from PIL import Image
except Exception:
    Image = None


def load_glove_embeddings(glove_path: str) -> Dict[str, np.ndarray]:
    """Load GloVe vectors from a text file into a dict: word -> numpy array.

    glove_path: path to a file like glove.6B.300d.txt
    Returns a dict mapping lowercased words to float32 numpy arrays.
    """
    emb = {}
    glove_path = Path(glove_path)
    if not glove_path.exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            emb[word.lower()] = vec

    return emb


class TeacherDataset(Dataset):
    """Dataset mapping text definitions + images to target word vectors (GloVe).

    Expects `dataset.json` (list of dicts) with at least 'word', 'definition', and 'image_dir' fields.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG"}

    def __init__(
        self,
        json_path: str,
        glove: Dict[str, np.ndarray],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        skip_missing: bool = True,
    ):
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed. Install via `pip install transformers`.")
        if Image is None:
            raise RuntimeError("Pillow not installed. Install via `pip install pillow`.")

        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.examples = []
        for it in items:
            word = it.get("word")
            definition = it.get("definition")
            image_dir = it.get("image_dir")
            if word is None or definition is None or image_dir is None:
                continue

            # Find image files in image_dir
            image_dir_path = Path(image_dir)
            image_files = []
            if image_dir_path.exists() and image_dir_path.is_dir():
                for p in sorted(image_dir_path.iterdir()):
                    if p.suffix in self.IMAGE_EXTS:
                        image_files.append(str(p))

            if not image_files and skip_missing:
                continue

            # Find glove vector (try lowercased exact match, then first token, then average tokens)
            vec = glove.get(word.lower())
            if vec is None:
                # Try splitting on space/underscore and averaging tokens
                import re
                tokens = [t for t in re.split(r"[\s_]+", word.lower()) if t]
                token_vecs = [glove.get(t) for t in tokens if glove.get(t) is not None]
                if token_vecs:
                    vec = np.mean(token_vecs, axis=0).astype(np.float32)

            if vec is None and skip_missing:
                continue

            # Fallback: random vector with same dim
            if vec is None:
                dim = next(iter(glove.values())).shape[0]
                vec = np.random.normal(scale=0.01, size=(dim,)).astype(np.float32)

            self.examples.append({
                "definition": definition,
                "word": word,
                "vector": vec,
                "image_files": image_files if image_files else [None],  # at least one None placeholder
            })

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Image transform
        if tv_transforms is not None:
            self.image_transform = tv_transforms.Compose([
                tv_transforms.Resize(256),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.image_transform = None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize definition
        toks = self.tokenizer(
            ex["definition"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Load image (deterministic selection by index modulo number of files)
        image_files = ex["image_files"]
        if image_files[0] is None:
            # Placeholder: return zeros (will be handled as "no image" case)
            image_tensor = torch.zeros(3, 224, 224)
        else:
            file_path = image_files[idx % len(image_files)]
            try:
                img = Image.open(file_path).convert("RGB")
                if self.image_transform:
                    image_tensor = self.image_transform(img)
                else:
                    # Fallback: convert to tensor and normalize shape
                    image_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            except Exception:
                # Fallback: zeros if image load fails
                image_tensor = torch.zeros(3, 224, 224)

        item = {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "image": image_tensor,
            "vector": torch.from_numpy(ex["vector"]).float(),
            "word": ex["word"],
        }
        return item


class TeacherModel(nn.Module):
    """Multimodal teacher model: BERT text + ResNet image -> target embedding vector.

    Args:
      bert_model_name: HuggingFace model id for text encoder (default: bert-base-uncased)
      target_dim: dimension of target vectors (e.g., 300 for GloVe)
      freeze_bert: if True, do not update BERT parameters during training
      modality_dropout_p: probability of zeroing out image during training (0.0 = no dropout, 0.2 = 20% dropout)
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        target_dim: int = 300,
        freeze_bert: bool = False,
        modality_dropout_p: float = 0.2,
    ):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers not installed. Install via `pip install transformers`.")
        if models is None:
            raise RuntimeError("torchvision not installed. Install via `pip install torchvision`.")

        self.modality_dropout_p = modality_dropout_p

        # Text encoder: BERT
        self.bert = AutoModel.from_pretrained(bert_model_name)
        text_hidden_size = self.bert.config.hidden_size
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # Image encoder: ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        # Remove the classification head
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # output: (batch, 2048, 1, 1)
        image_hidden_size = 2048

        # Fusion: concatenate text + image -> linear -> target_dim
        fused_dim = text_hidden_size + image_hidden_size
        self.fusion_head = nn.Linear(fused_dim, target_dim)

    def forward(self, input_ids, attention_mask, image, training: bool = False):
        """Forward pass.

        Args:
          input_ids: (batch, seq_len)
          attention_mask: (batch, seq_len)
          image: (batch, 3, 224, 224)
          training: if True, apply modality dropout

        Returns:
          (batch, target_dim) prediction vector
        """
        # Text encoding
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_feat = text_outputs.pooler_output  # (batch, hidden_size)
        else:
            text_feat = text_outputs.last_hidden_state[:, 0, :]  # use CLS token

        # Image encoding
        image_feat = self.resnet(image)  # (batch, 2048, 1, 1)
        image_feat = image_feat.view(image_feat.size(0), -1)  # (batch, 2048)

        # Modality dropout: randomly zero out image during training
        if training and self.modality_dropout_p > 0:
            if torch.rand(1).item() < self.modality_dropout_p:
                image_feat = torch.zeros_like(image_feat)

        # Fusion
        fused = torch.cat([text_feat, image_feat], dim=1)  # (batch, text_hidden + image_hidden)
        pred = self.fusion_head(fused)  # (batch, target_dim)
        return pred


def train_one_epoch(model, dataloader, optimizer, device, criterion=None):
    model.train()
    if criterion is None:
        criterion = nn.MSELoss()

    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        target = batch["vector"].to(device)

        optimizer.zero_grad()
        pred = model(input_ids=input_ids, attention_mask=attention_mask, image=image, training=True)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            target = batch["vector"].to(device)

            pred = model(input_ids=input_ids, attention_mask=attention_mask, image=image, training=False)
            loss = criterion(pred, target)
            total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


def collate_examples(batch):
    """Collate batch of examples from TeacherDataset."""
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    images = torch.stack([b["image"] for b in batch], dim=0)
    vectors = torch.stack([b["vector"] for b in batch], dim=0)
    words = [b.get("word") for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image": images,
        "vector": vectors,
        "words": words,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the multimodal Teacher model")
    parser.add_argument("--dataset", default="dataset.json", help="Path to dataset.json")
    parser.add_argument("--glove", required=True, help="Path to GloVe txt file (e.g., glove.6B.300d.txt)")
    parser.add_argument("--bert", default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--maxlen", type=int, default=128, help="Max token length for BERT")
    parser.add_argument("--modality-dropout", type=float, default=0.2, help="Modality dropout probability (0.0-1.0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu/mps)")
    parser.add_argument("--output", default="./ckpts/teacher_model.pt", help="Output model path")
    args = parser.parse_args()

    print("Loading GloVe... this may take a while")
    glove = load_glove_embeddings(args.glove)
    sample_dim = next(iter(glove.values())).shape[0]
    print(f"GloVe dim: {sample_dim}")

    print("Loading dataset...")
    ds = TeacherDataset(args.dataset, glove, tokenizer_name=args.bert, max_length=args.maxlen)
    n = len(ds)
    print(f"Dataset size: {n}")

    # Simple train/val split
    train_n = int(0.9 * n)
    indices = list(range(n))
    train_ds = torch.utils.data.Subset(ds, indices[:train_n])
    val_ds = torch.utils.data.Subset(ds, indices[train_n:])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=lambda x: collate_examples(x))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=lambda x: collate_examples(x))

    device = torch.device(args.device)
    model = TeacherModel(
        bert_model_name=args.bert,
        target_dim=sample_dim,
        modality_dropout_p=args.modality_dropout,
    )
    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Training for {args.epochs} epochs with modality_dropout={args.modality_dropout}...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
