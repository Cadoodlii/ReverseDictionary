"""Visualize embeddings comparison for k random hypernym groups.

This script visualizes four embedding spaces side-by-side for the same random hypernym groups:
1. StudentModel (text-only) embeddings
2. DistilledModel (text-only, trained via knowledge distillation) embeddings
3. TeacherModel (multimodal with zero'd images) embeddings
4. GloVe (ground-truth) embeddings

Each visualization applies t-SNE to reduce 300-dim embeddings to 2D, with points colored by hypernym group.
This allows direct comparison of how well each model learns semantic clustering.

Requirements:
  pip install numpy torch transformers scikit-learn matplotlib pillow tqdm torchvision

How to run:
# Basic usage (5 random hypernyms, CPU)
python3 visualize_embeddings.py

# With custom options
python3 visualize_embeddings.py --k 8 --device cpu --perplexity 20 --output comparison_viz.png

# With GPU
python3 visualize_embeddings.py --k 5 --device cuda
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from student import load_glove_embeddings, TextOnlyDataset, StudentModel
from teacher import TeacherDataset, TeacherModel


def group_by_hypernym(dataset_json: str) -> Dict[str, List[Dict]]:
    """Group dataset entries by hypernym."""
    with open(dataset_json, 'r') as f:
        data = json.load(f)
    
    groups = {}
    for entry in data:
        hypernym = entry.get('hypernym', 'unknown')
        if hypernym not in groups:
            groups[hypernym] = []
        groups[hypernym].append(entry)
    
    return groups


def get_model_embeddings(
    entries: List[Dict],
    model: torch.nn.Module,
    ds,
    device: torch.device,
    model_type: str = 'student',
    use_zero_images: bool = False,
) -> np.ndarray:
    """Get model embeddings for a list of entries.
    
    Args:
        entries: list of dataset entries with 'definition' (and 'image_dir' for teacher)
        model: torch.nn.Module (StudentModel or TeacherModel)
        ds: dataset instance (TextOnlyDataset or TeacherDataset)
        device: torch device
        model_type: 'student' or 'teacher'
        use_zero_images: if True (for teacher), use zero'd images instead of actual images
    
    Returns:
        Array of shape (len(entries), embedding_dim)
    """
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for entry in entries:
            definition = entry['definition']
            tokenizer = ds.tokenizer
            toks = tokenizer(
                definition,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            input_ids = toks['input_ids'].to(device)
            attention_mask = toks['attention_mask'].to(device)
            
            if model_type.lower() == 'student':
                pred = model(input_ids=input_ids, attention_mask=attention_mask)
            elif model_type.lower() == 'teacher':
                # For teacher model
                if use_zero_images:
                    # Use zero'd images
                    image = torch.zeros(1, 3, 224, 224).to(device)
                else:
                    # Try to load actual images
                    image_files = []
                    image_dir_path = Path(entry.get('image_dir', ''))
                    if image_dir_path.exists() and image_dir_path.is_dir():
                        for p in sorted(image_dir_path.iterdir()):
                            if p.suffix in {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG'}:
                                image_files.append(str(p))
                    
                    if image_files:
                        # Load first image for this entry
                        try:
                            from PIL import Image
                            img = Image.open(image_files[0]).convert('RGB')
                            if hasattr(ds, 'image_transform') and ds.image_transform is not None:
                                image = ds.image_transform(img).unsqueeze(0).to(device)
                            else:
                                image = torch.zeros(1, 3, 224, 224).to(device)
                        except Exception:
                            image = torch.zeros(1, 3, 224, 224).to(device)
                    else:
                        image = torch.zeros(1, 3, 224, 224).to(device)
                
                pred = model(input_ids=input_ids, attention_mask=attention_mask, image=image, training=False)
            
            embeddings.append(pred.cpu().numpy().reshape(-1))
    
    return np.array(embeddings)


def get_glove_embeddings(
    entries: List[Dict],
    glove: Dict[str, np.ndarray],
) -> np.ndarray:
    """Get GloVe embeddings for a list of entries (using target word).
    
    Returns:
        Array of shape (len(entries), embedding_dim)
    """
    embeddings = []
    for entry in entries:
        word = entry['word']
        # Try exact match, then first token
        vec = glove.get(word.lower())
        if vec is None:
            # Split on space/underscore and average tokens
            import re
            tokens = [t for t in re.split(r'[\s_]+', word.lower()) if t]
            token_vecs = [glove.get(t) for t in tokens if glove.get(t) is not None]
            if token_vecs:
                vec = np.mean(token_vecs, axis=0)
            else:
                # Fallback: random
                dim = next(iter(glove.values())).shape[0]
                vec = np.random.normal(scale=0.01, size=(dim,)).astype(np.float32)
        embeddings.append(vec)
    
    return np.array(embeddings)


def visualize_all_embeddings(
    dataset_json: str = 'dataset.json',
    glove_path: str = './glove.6B.300d.txt',
    student_model_path: str = './ckpts/student_model.pt',
    distilled_model_path: str = './ckpts/student_distilled_w_dropout.pt',
    teacher_model_path: str = './ckpts/teacher_model_w_dropout.pt',
    bert_model: str = 'bert-base-uncased',
    k_hypernyms: int = 5,
    perplexity: int = 30,
    device_name: str = 'cpu',
    output_path: str = 'embeddings_comparison.png',
):
    """Visualize StudentModel, DistilledModel, TeacherModel (with zero'd images), and GloVe embeddings side-by-side."""
    
    device = torch.device(device_name)
    print(f'Using device: {device}')
    
    # Load GloVe
    print('Loading GloVe...')
    glove = load_glove_embeddings(glove_path)
    embedding_dim = next(iter(glove.values())).shape[0]
    print(f'GloVe dimension: {embedding_dim}')
    
    # Load datasets
    print('Loading datasets...')
    student_ds = TextOnlyDataset(dataset_json, glove, tokenizer_name=bert_model, max_length=128)
    teacher_ds = TeacherDataset(dataset_json, glove, tokenizer_name=bert_model, max_length=128)
    
    # Load models
    print('Loading models...')
    student_model = StudentModel(bert_model_name=bert_model, target_dim=embedding_dim)
    distilled_model = StudentModel(bert_model_name=bert_model, target_dim=embedding_dim)
    teacher_model = TeacherModel(bert_model_name=bert_model, target_dim=embedding_dim)
    
    if Path(student_model_path).exists():
        try:
            student_model.load_state_dict(torch.load(student_model_path, map_location=device))
            print(f'Loaded StudentModel from {student_model_path}')
        except Exception as e:
            print(f'Warning: could not load StudentModel weights: {e}')
    else:
        print(f'Warning: StudentModel not found at {student_model_path}, using random initialization')
    
    if Path(distilled_model_path).exists():
        try:
            distilled_model.load_state_dict(torch.load(distilled_model_path, map_location=device))
            print(f'Loaded DistilledModel from {distilled_model_path}')
        except Exception as e:
            print(f'Warning: could not load DistilledModel weights: {e}')
    else:
        print(f'Warning: DistilledModel not found at {distilled_model_path}, using random initialization')
    
    if Path(teacher_model_path).exists():
        try:
            teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
            print(f'Loaded TeacherModel from {teacher_model_path}')
        except Exception as e:
            print(f'Warning: could not load TeacherModel weights: {e}')
    else:
        print(f'Warning: TeacherModel not found at {teacher_model_path}, using random initialization')
    
    student_model.to(device)
    distilled_model.to(device)
    teacher_model.to(device)
    
    # Group by hypernym
    print('Grouping by hypernym...')
    groups = group_by_hypernym(dataset_json)
    print(f'Total hypernym groups: {len(groups)}')
    
    # Pick k random hypernyms with enough entries
    min_entries = 3
    valid_hypernyms = [h for h, entries in groups.items() if len(entries) >= min_entries]
    print(f'Valid hypernyms (>= {min_entries} entries): {len(valid_hypernyms)}')
    
    if len(valid_hypernyms) < k_hypernyms:
        k_hypernyms = len(valid_hypernyms)
        print(f'Reducing k to {k_hypernyms} (not enough valid hypernyms)')
    
    selected_hypernyms = random.sample(valid_hypernyms, k_hypernyms)
    print(f'Selected {k_hypernyms} random hypernyms: {selected_hypernyms}')
    
    # Collect all entries for selected hypernyms
    all_entries = []
    hypernym_labels = []
    for hypernym in selected_hypernyms:
        entries = groups[hypernym]
        all_entries.extend(entries)
        hypernym_labels.extend([hypernym] * len(entries))
    
    print(f'Total words to visualize: {len(all_entries)}')
    
    # Get embeddings from all four sources
    print('Computing StudentModel embeddings...')
    student_embeddings = get_model_embeddings(all_entries, student_model, student_ds, device, model_type='student')
    print(f'StudentModel embeddings shape: {student_embeddings.shape}')
    
    print('Computing DistilledModel embeddings...')
    distilled_embeddings = get_model_embeddings(all_entries, distilled_model, student_ds, device, model_type='student')
    print(f'DistilledModel embeddings shape: {distilled_embeddings.shape}')
    
    print('Computing TeacherModel embeddings (with zero\'d images)...')
    teacher_embeddings = get_model_embeddings(all_entries, teacher_model, teacher_ds, device, model_type='teacher', use_zero_images=True)
    print(f'TeacherModel embeddings shape: {teacher_embeddings.shape}')
    
    print('Getting GloVe embeddings...')
    glove_embeddings = get_glove_embeddings(all_entries, glove)
    print(f'GloVe embeddings shape: {glove_embeddings.shape}')
    
    # Apply t-SNE (fit on all embeddings combined for consistency)
    print(f'Applying t-SNE (perplexity={perplexity}) on combined embeddings...')
    combined = np.vstack([student_embeddings, distilled_embeddings, teacher_embeddings, glove_embeddings])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    combined_2d = tsne.fit_transform(combined)
    
    n = len(all_entries)
    student_2d = combined_2d[:n]
    distilled_2d = combined_2d[n:2*n]
    teacher_2d = combined_2d[2*n:3*n]
    glove_2d = combined_2d[3*n:]
    
    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, k_hypernyms))
    hypernym_to_color = {h: colors[i] for i, h in enumerate(selected_hypernyms)}
    point_colors = [hypernym_to_color[h] for h in hypernym_labels]
    
    # Plot all four side-by-side
    fig, axes = plt.subplots(1, 4, figsize=(32, 7))
    
    # StudentModel embeddings
    axes[0].scatter(student_2d[:, 0], student_2d[:, 1], c=point_colors, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].set_title(f'Student Model (Text-Only)\n{k_hypernyms} hypernym groups', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE dimension 1')
    axes[0].set_ylabel('t-SNE dimension 2')
    axes[0].grid(True, alpha=0.3)
    
    # DistilledModel embeddings
    axes[1].scatter(distilled_2d[:, 0], distilled_2d[:, 1], c=point_colors, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].set_title(f'Distilled Model (Text-Only)\n{k_hypernyms} hypernym groups', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE dimension 1')
    axes[1].set_ylabel('t-SNE dimension 2')
    axes[1].grid(True, alpha=0.3)
    
    # TeacherModel embeddings (with zero'd images)
    axes[2].scatter(teacher_2d[:, 0], teacher_2d[:, 1], c=point_colors, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[2].set_title(f'Teacher Model (Zero\'d Images)\n{k_hypernyms} hypernym groups', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('t-SNE dimension 1')
    axes[2].set_ylabel('t-SNE dimension 2')
    axes[2].grid(True, alpha=0.3)
    
    # GloVe embeddings
    axes[3].scatter(glove_2d[:, 0], glove_2d[:, 1], c=point_colors, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[3].set_title(f'GloVe (Ground Truth)\n{k_hypernyms} hypernym groups', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('t-SNE dimension 1')
    axes[3].set_ylabel('t-SNE dimension 2')
    axes[3].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=hypernym_to_color[h], edgecolor='k', label=h) for h in selected_hypernyms]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=min(5, k_hypernyms))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved visualization to {output_path}')
    plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize student, distilled, teacher (zero\'d images), and GloVe embeddings side-by-side for hypernym groups')
    parser.add_argument('--dataset', default='dataset.json', help='Path to dataset.json')
    parser.add_argument('--glove', default='./glove.6B.300d.txt', help='Path to GloVe file')
    parser.add_argument('--student-model', default='./ckpts/student_model.pt', help='Path to student model checkpoint')
    parser.add_argument('--distilled-model', default='./ckpts/student_distilled_w_dropout.pt', help='Path to distilled model checkpoint')
    parser.add_argument('--teacher-model', default='./ckpts/teacher_model_w_dropout.pt', help='Path to teacher model checkpoint')
    parser.add_argument('--bert', default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--k', type=int, default=5, help='Number of random hypernyms to visualize')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--device', default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--output', default='embeddings_comparison.png', help='Output image path')
    
    args = parser.parse_args()
    
    visualize_all_embeddings(
        dataset_json=args.dataset,
        glove_path=args.glove,
        student_model_path=args.student_model,
        distilled_model_path=args.distilled_model,
        teacher_model_path=args.teacher_model,
        bert_model=args.bert,
        k_hypernyms=args.k,
        perplexity=args.perplexity,
        device_name=args.device,
        output_path=args.output,
    )
