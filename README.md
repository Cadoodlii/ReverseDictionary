
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
- `teacher_test.ipynb`: Notebook that trains the `TeacherModel` with multimodal input, performs retrieval, and computes recall@k.
- `visualize_embeddings.py`: Script to create unified side-by-side visualizations of StudentModel, TeacherModel, and GloVe embeddings using t-SNE for selected hypernym groups.

**StudentModel (text-only baseline)**
- Architecture: BERT encoder (HuggingFace `AutoModel`) → pooled representation → linear head to predict a target embedding vector (e.g., 300-dim for GloVe).
- Targets: GloVe vectors for the target word. For multi-token target words (e.g., `electric_guitar` or `red apple`) the target vector is the average of the GloVe vectors for each token (split on whitespace/underscore).
- Loss: Mean Squared Error (MSE) between predicted vector and the ground-truth GloVe vector.
- Files:
	- `student.py` contains `load_glove_embeddings`, `TextOnlyDataset`, `StudentModel`, `train_one_epoch`, `evaluate`, and a CLI for training.
	- `student_model.pt` (optional): saved model checkpoint produced by training (if present).

**TeacherModel (multimodal - RDMIF-style)**
- Architecture:
  - Text Encoder: BERT (same as StudentModel)
  - Image Encoder: ResNet-50 (pre-trained on ImageNet)
  - Fusion: Concatenate text + image features → Linear layer → target embedding (300-dim for GloVe)
- Co-Learning (Modality Dropout): During training, randomly zero out image features with probability `modality_dropout_p` (default 0.2). This forces the model to rely on text alone during some iterations, encouraging balanced multimodal learning.
- Loss: MSELoss between predicted vector and ground-truth GloVe vector (same as StudentModel).
- Files:
  - `teacher.py` contains `TeacherModel`, `TeacherDataset`, training/eval utilities, and a CLI for training.
  - `teacher_model.pt` (optional): saved model checkpoint (if present).

**Retrieval & Evaluation**
- Retrieval is done by encoding a definition (+ optionally an image for the teacher) with the trained model to produce a query vector, then ranking dataset words by cosine similarity between the query and the dataset GloVe vectors.
- `student_test.ipynb` includes a `retrieve_topk(definition, k)` helper and computes recall@1, recall@5, and recall@10 on a held-out test split.

**Embedding Visualization**
- `visualize_embeddings.py` creates unified side-by-side visualizations comparing three embedding spaces:
  - **StudentModel**: Text-only embeddings (left subplot)
  - **TeacherModel**: Multimodal (text + image) embeddings (center subplot)
  - **GloVe**: Ground-truth embeddings from GloVe (right subplot)
- Visualizations use t-SNE for 2D dimensionality reduction and color hypernym groups for semantic structure.
- Command: `python3 visualize_embeddings.py --k 5 --device cpu --output comparison.png`
- The same random hypernyms and consistent t-SNE fit are used across all three plots for fair comparison.
- Optional arguments:
  - `--dataset`: path to dataset.json (default: `dataset.json`)
  - `--glove`: path to GloVe embeddings (default: `./glove.6B.300d.txt`)
  - `--student-model`: path to student model checkpoint (default: `./ckpts/student_model.pt`)
  - `--teacher-model`: path to teacher model checkpoint (default: `./ckpts/teacher_model.pt`)
  - `--k`: number of random hypernym groups to visualize (default: 5)
  - `--perplexity`: t-SNE perplexity parameter (default: 30)
  - `--device`: device for model inference, options: `cpu`, `cuda`, `mps` (default: `cpu`)
  - `--output`: output image path (default: `embeddings_comparison.png`)

**Quick start**
1. Install dependencies:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy torch transformers tqdm pillow torchvision
```
2. Ensure you have a GloVe file (e.g., `glove.6B.300d.txt`) in the repo root.
3. Train the **StudentModel** (text-only baseline):
```bash
python3 student.py --dataset dataset.json --glove ./glove.6B.300d.txt --epochs 3 --batch 16
```
4. Train the **TeacherModel** (multimodal with text + image):
```bash
python3 teacher.py --dataset dataset.json --glove ./glove.6B.300d.txt --epochs 3 --batch 8 --modality-dropout 0.2
```
5. Run retrieval and evaluation in a notebook:
   - Open `student_test.ipynb` (Jupyter) and run cells sequentially
   - Modify cells to use `TeacherModel` if desired

**Example: Using TeacherModel in code**
```python
from teacher import TeacherModel, TeacherDataset, load_glove_embeddings
import torch

glove = load_glove_embeddings('./glove.6B.300d.txt')
ds = TeacherDataset('dataset.json', glove)
model = TeacherModel(target_dim=300, modality_dropout_p=0.2)

# Forward pass (training=True enables modality dropout)
sample = ds[0]
input_ids = sample['input_ids'].unsqueeze(0)
attention_mask = sample['attention_mask'].unsqueeze(0)
image = sample['image'].unsqueeze(0)
output = model(input_ids, attention_mask, image, training=True)  # shape: (1, 300)
```