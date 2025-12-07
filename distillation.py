"""Knowledge Distillation for Reverse Dictionary.

Concept: Train a Text-Only StudentModel to mimic a Multimodal TeacherModel.

The distillation loop:
1. Pass (Definition, Image) to frozen TeacherModel -> Get Teacher_Vector
2. Pass (Definition) to StudentModel -> Get Student_Vector
3. Loss: Minimize distance (MSE or KL) between Student_Vector and Teacher_Vector

This teaches the student to produce the same embeddings as the teacher,
effectively allowing the text-only student to benefit from the teacher's
multimodal understanding without needing images at inference time.

Requirements:
  pip install torch transformers numpy tqdm torchvision pillow scikit-learn

Example usage:
  python3 distillation.py \\
    --dataset dataset.json \\
    --glove ./glove.6B.300d.txt \\
    --teacher-model ./ckpts/teacher_model.pt \\
    --epochs 15 \\
    --batch 16 \\
    --lr 2e-5 \\
    --output ./ckpts/student_distilled.pt

"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Lazy imports
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

# Import our models
from teacher import TeacherModel, TeacherDataset, load_glove_embeddings
from student import StudentModel, TextOnlyDataset


def create_distillation_dataset(
    dataset_json: str,
    glove_dict: Dict[str, np.ndarray],
    tokenizer_name: str = 'bert-base-uncased',
    max_length: int = 128,
):
    """Create a dataset that provides (definition, image, ground_truth_glove_vector).
    
    For distillation, we primarily need definitions and images + their indices.
    The ground truth GloVe vector is used as a fallback target if teacher output is not available.
    
    Returns:
        ds: TeacherDataset (which includes both text and image)
    """
    return TeacherDataset(dataset_json, glove_dict, tokenizer_name=tokenizer_name, max_length=max_length)


def train_distillation_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataset: Dataset,
    batch_size: int = 16,
    device: torch.device = torch.device('cpu'),
    lr: float = 2e-5,
    temperature: float = 1.0,
    alpha: float = 0.5,
):
    """Train student for one epoch using hybrid loss (teacher + ground truth).
    
    Hybrid Loss: Loss = alpha * MSE(student, teacher) + (1-alpha) * MSE(student, glove)
    
    Args:
        student_model: Text-only StudentModel (trainable)
        teacher_model: Multimodal TeacherModel (frozen)
        dataset: TeacherDataset with (def, image, gt_vector)
        batch_size: Batch size
        device: Device to use
        lr: Learning rate (for optimizer creation)
        temperature: Temperature for soft targets (reserved for future use)
        alpha: Weight for teacher loss (default 0.5). Loss = alpha * teacher + (1-alpha) * glove
    
    Returns:
        avg_loss: Average loss over the epoch
    """
    teacher_model.eval()  # Frozen teacher
    student_model.train()
    
    # Create data loader
    from teacher import collate_examples  # Use teacher's collate which handles images
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_examples(x),
    )
    
    optimizer = torch.optim.AdamW([p for p in student_model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc='Distillation epoch', leave=False)
    for batch in pbar:
        # Extract batch data
        # batch is a dict with keys: input_ids, attention_mask, image, vector, word
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image = batch['image'].to(device)
        gt_vector = batch['vector'].to(device)  # Ground truth GloVe vectors
        
        # Forward through frozen teacher (text + image)
        with torch.no_grad():
            teacher_output = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                training=False
            )
        
        # Forward through student (text only)
        student_output = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Hybrid loss: alpha * teacher_loss + (1-alpha) * glove_loss
        teacher_loss = criterion(student_output, teacher_output.detach())
        glove_loss = criterion(student_output, gt_vector)
        loss = alpha * teacher_loss + (1.0 - alpha) * glove_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item(), 'alpha': alpha})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataset: Dataset,
    batch_size: int = 16,
    device: torch.device = torch.device('cpu'),
    alpha: float = 0.5,
):
    """Evaluate distillation on a validation set using hybrid loss.
    
    Computes:
    - Hybrid MSE (alpha * teacher + (1-alpha) * glove)
    - Teacher-specific MSE
    - GloVe-specific MSE
    - Cosine similarity between student and teacher outputs
    
    Args:
        student_model: Text-only StudentModel
        teacher_model: Multimodal TeacherModel
        dataset: Validation dataset
        batch_size: Batch size
        device: Device to use
        alpha: Weight for teacher loss in hybrid computation
    
    Returns:
        avg_hybrid_mse: Average hybrid MSE
        avg_teacher_mse: Average teacher-specific MSE
        avg_glove_mse: Average GloVe-specific MSE
        avg_cosine: Average cosine similarity between student and teacher
    """
    teacher_model.eval()
    student_model.eval()
    
    from teacher import collate_examples  # Use teacher's collate which handles images
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_examples(x),
    )
    
    criterion = nn.MSELoss()
    total_hybrid_mse = 0.0
    total_teacher_mse = 0.0
    total_glove_mse = 0.0
    total_cosine = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc='Distillation eval', leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image = batch['image'].to(device)
        gt_vector = batch['vector'].to(device)
        
        with torch.no_grad():
            teacher_output = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                training=False
            )
            student_output = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # MSE components
            teacher_mse = criterion(student_output, teacher_output)
            glove_mse = criterion(student_output, gt_vector)
            hybrid_mse = alpha * teacher_mse + (1.0 - alpha) * glove_mse
            
            total_hybrid_mse += hybrid_mse.item()
            total_teacher_mse += teacher_mse.item()
            total_glove_mse += glove_mse.item()
            
            # Cosine similarity (per sample in batch)
            student_np = student_output.cpu().numpy()
            teacher_np = teacher_output.cpu().numpy()
            batch_cosine = np.mean([
                np.dot(student_np[i], teacher_np[i]) / 
                (np.linalg.norm(student_np[i]) * np.linalg.norm(teacher_np[i]) + 1e-8)
                for i in range(len(student_np))
            ])
            total_cosine += batch_cosine
            
            num_batches += 1
    
    avg_hybrid_mse = total_hybrid_mse / max(num_batches, 1)
    avg_teacher_mse = total_teacher_mse / max(num_batches, 1)
    avg_glove_mse = total_glove_mse / max(num_batches, 1)
    avg_cosine = total_cosine / max(num_batches, 1)
    
    return avg_hybrid_mse, avg_teacher_mse, avg_glove_mse, avg_cosine


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge distillation: Train text-only student to mimic multimodal teacher.'
    )
    parser.add_argument('--dataset', type=str, default='dataset.json', help='Path to dataset.json')
    parser.add_argument('--glove', type=str, default='./glove.6B.300d.txt', help='Path to GloVe file')
    parser.add_argument('--teacher-model', type=str, default='./ckpts/teacher_model.pt', 
                        help='Path to pre-trained teacher model checkpoint')
    parser.add_argument('--student-model-init', type=str, default=None,
                        help='Path to initialize student model (optional)')
    parser.add_argument('--bert', type=str, default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--maxlen', type=int, default=128, help='Max token length')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/mps)')
    parser.add_argument('--output', type=str, default='./ckpts/student_distilled.pt',
                        help='Output path for distilled student model')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--temperature', type=float, default=1.0, help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Weight for teacher loss (0.5 = 50%% teacher + 50%% glove)')
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')
    
    # Load GloVe
    print('Loading GloVe...')
    glove = load_glove_embeddings(args.glove)
    embedding_dim = next(iter(glove.values())).shape[0]
    print(f'GloVe dimension: {embedding_dim}')
    
    # Load dataset (TeacherDataset with images)
    print('Loading dataset...')
    ds = create_distillation_dataset(args.dataset, glove, tokenizer_name=args.bert, max_length=args.maxlen)
    print(f'Dataset size: {len(ds)}')
    
    # Train/val split
    n = len(ds)
    val_n = int(args.val_split * n)
    train_n = n - val_n
    
    import random
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]
    
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    
    print(f'Train size: {len(train_ds)}, Val size: {len(val_ds)}')
    
    # Create models
    print('Creating models...')
    
    # Teacher (load pre-trained, freeze)
    teacher = TeacherModel(bert_model_name=args.bert, target_dim=embedding_dim)
    if Path(args.teacher_model).exists():
        try:
            teacher.load_state_dict(torch.load(args.teacher_model, map_location=device))
            print(f'Loaded teacher from {args.teacher_model}')
        except Exception as e:
            print(f'Warning: Could not load teacher: {e}')
    else:
        print(f'Warning: Teacher model not found at {args.teacher_model}, using random init')
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Student (text-only, trainable)
    student = StudentModel(bert_model_name=args.bert, target_dim=embedding_dim)
    if args.student_model_init and Path(args.student_model_init).exists():
        try:
            student.load_state_dict(torch.load(args.student_model_init, map_location=device))
            print(f'Initialized student from {args.student_model_init}')
        except Exception as e:
            print(f'Warning: Could not initialize student: {e}')
    else:
        print('Student initialized with random weights')
    student.to(device)
    
    print(f'\nStarting distillation training for {args.epochs} epochs (alpha={args.alpha})...\n')
    
    # Training loop
    best_val_mse = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_distillation_epoch(
            student, teacher, train_ds,
            batch_size=args.batch,
            device=device,
            lr=args.lr,
            temperature=args.temperature,
            alpha=args.alpha,
        )
        
        val_hybrid_mse, val_teacher_mse, val_glove_mse, val_cosine = evaluate_distillation(
            student, teacher, val_ds,
            batch_size=args.batch,
            device=device,
            alpha=args.alpha,
        )
        
        print(f'Epoch {epoch}: train_loss={train_loss:.6f} val_hybrid_mse={val_hybrid_mse:.6f} val_teacher_mse={val_teacher_mse:.6f} val_glove_mse={val_glove_mse:.6f} val_cosine={val_cosine:.6f}')
        
        # Save best model based on hybrid MSE
        if val_hybrid_mse < best_val_mse:
            best_val_mse = val_hybrid_mse
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), output_path)
            print(f'Saved best model to {output_path}')
    
    print(f'\nDistillation complete. Best model saved to {args.output}')


if __name__ == '__main__':
    main()
