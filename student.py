"""Student model for Text-Only Reverse Dictionary baseline.

Architecture:
- BERT-base encoder -> pooled token -> Linear -> target embedding dim (e.g., 300)

Loss: MSELoss between predicted vector and ground truth GloVe vector.

Utilities included:
- load_glove_embeddings(glove_path)
- TextOnlyDataset to pair definitions -> glove vector
- StudentModel (torch.nn.Module)
- train/evaluate helper functions and CLI example

Requirements:
  pip install torch transformers numpy tqdm

Run example (after preparing `dataset.json` and GloVe file):
  python3 student.py --dataset dataset.json --glove /path/to/glove.6B.300d.txt --epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None


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


class TextOnlyDataset(Dataset):
    """Dataset mapping text definitions to target word vectors (GloVe).

    Expects `dataset.json` (list of dicts) with at least 'word' and 'definition' fields.
    """

    def __init__(
        self,
        json_path: str,
        glove: Dict[str, np.ndarray],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        skip_missing: bool = True,
    ):
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed. Install via `pip install transformers` to use TextOnlyDataset.`")

        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.examples = []
        for it in items:
            word = it.get("word")
            definition = it.get("definition")
            if word is None or definition is None:
                continue

            # Compute the target vector as the average of GloVe vectors
            # for each token in the target `word`. Tokens are split on
            # whitespace and underscores (e.g., "electric_guitar" -> ["electric","guitar"]).
            tokens = [t for t in re.split(r"[\s_]+", word.lower()) if t]
            vec = None
            if tokens:
                token_vecs = [glove.get(t) for t in tokens if glove.get(t) is not None]
                if token_vecs:
                    vec = np.mean(token_vecs, axis=0).astype(np.float32)

            if vec is None and skip_missing:
                continue

            # fallback: random vector with same dim
            if vec is None:
                dim = next(iter(glove.values())).shape[0]
                vec = np.random.normal(scale=0.01, size=(dim,)).astype(np.float32)

            self.examples.append({"definition": definition, "word": word, "vector": vec})

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        toks = self.tokenizer(
            ex["definition"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "vector": torch.from_numpy(ex["vector"]).float(),
            "word": ex["word"],
        }
        return item


class StudentModel(nn.Module):
    """BERT-based student model that maps a definition to a target embedding vector.

    Args:
      bert_model_name: HuggingFace model id for the encoder
      target_dim: dimension of target vectors (e.g., 300 for GloVe)
      freeze_bert: if True, do not update BERT parameters during training
    """

    def __init__(self, bert_model_name: str = "bert-base-uncased", target_dim: int = 300, freeze_bert: bool = False):
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers not installed. Install via `pip install transformers` to use StudentModel.")

        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.head = nn.Linear(hidden_size, target_dim)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # Use pooled output when available, else use CLS token hidden state
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = None
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # use CLS token
            pooled = outputs.last_hidden_state[:, 0, :]

        pred = self.head(pooled)
        return pred


def train_one_epoch(model, dataloader, optimizer, device, criterion=None):
    model.train()
    if criterion is None:
        criterion = nn.MSELoss()

    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["vector"].to(device)

        optimizer.zero_grad()
        pred = model(input_ids=input_ids, attention_mask=attention_mask)
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
            target = batch["vector"].to(device)

            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(pred, target)
            total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--glove", required=True, help="Path to GloVe txt file (e.g., glove.6B.300d.txt)")
    parser.add_argument("--bert", default="bert-base-uncased")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading GloVe... this may take a while")
    glove = load_glove_embeddings(args.glove)
    sample_dim = next(iter(glove.values())).shape[0]
    print(f"GloVe dim: {sample_dim}")

    ds = TextOnlyDataset(args.dataset, glove, tokenizer_name=args.bert, max_length=args.maxlen)
    n = len(ds)
    print(f"Dataset size: {n}")

    # simple train/val split
    train_n = int(0.9 * n)
    indices = list(range(n))
    train_ds = torch.utils.data.Subset(ds, indices[:train_n])
    val_ds = torch.utils.data.Subset(ds, indices[train_n:])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=lambda x: collate_examples(x))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=lambda x: collate_examples(x))

    device = torch.device(args.device)
    model = StudentModel(bert_model_name=args.bert, target_dim=sample_dim)
    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")


def collate_examples(batch):
    # batch is list of dicts from TextOnlyDataset.__getitem__
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    vectors = torch.stack([b["vector"] for b in batch], dim=0)
    words = [b.get("word") for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "vector": vectors, "words": words}


if __name__ == "__main__":
    main()
