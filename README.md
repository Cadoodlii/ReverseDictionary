
# ReverseDictionary — Repo Overview

This repository contains code and data for a Reverse Dictionary-style task using ImageNet synset labels and textual definitions.

**Layout**
- `dataset.json`: Starter dataset (list of items). Each item is a JSON object with at least these keys:
	- `word`: the synset label / target word (string)
	- `definition`: a text definition for the word (string)
	- `synset_id`: ImageNet wnid (e.g., `n01440764`) (string)
	- `image_dir`: path to the ImageNet validation directory for that synset (string)
- `data/LOC_synset_mappings.txt`: mapping from ImageNet wnid to human-readable label(s).
- `data/imagenet-val/`: ImageNet-1k validation images organized by wnid folders (each contains multiple `ILSVRC2012_val_*.JPEG` image files).
- `glove.6B.*.txt`: GloVe embedding files (e.g., `glove.6B.300d.txt`) used as target vectors.
- `student.py`: Text-only baseline implementation (dataset, model, training/eval utilities).
- `dataset.py`: Pseudo RDMIF dataset builder and `PseudoRDMIFDataset` class for image+definition loading.
- `examples/`: small example scripts (e.g., `examples/test_dataset.py`) demonstrating dataset loading and visualization.
- `student_test.ipynb`: Notebook that trains the text-only `StudentModel`, performs retrieval, and computes recall@k.

**StudentModel (text-only baseline)**
- Architecture: BERT encoder (HuggingFace `AutoModel`) → pooled representation → linear head to predict a target embedding vector (e.g., 300-dim for GloVe).
- Targets: GloVe vectors for the target word. For multi-token target words (e.g., `electric_guitar` or `red apple`) the target vector is the average of the GloVe vectors for each token (split on whitespace/underscore).
- Loss: Mean Squared Error (MSE) between predicted vector and the ground-truth GloVe vector.
- Files:
	- `student.py` contains `load_glove_embeddings`, `TextOnlyDataset`, `StudentModel`, `train_one_epoch`, `evaluate`, and a CLI for training.
	- `student_model.pt` (optional): saved model checkpoint produced by training (if present).

**Retrieval & Evaluation**
- Retrieval is done by encoding a definition with the trained `StudentModel` to produce a query vector, then ranking dataset words by cosine similarity between the query and the dataset GloVe vectors.
- `student_test.ipynb` includes a `retrieve_topk(definition, k)` helper and computes recall@1, recall@5, and recall@10 on a held-out test split.

**Quick start**
1. Install dependencies:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy torch transformers tqdm pillow
```
2. Ensure you have a GloVe file (e.g., `glove.6B.300d.txt`) and ImageNet files in the repo root (look at .gitignore for additional files).
3. Run the notebook `student_test.ipynb` (Jupyter) or run `python3 student.py` for CLI training.