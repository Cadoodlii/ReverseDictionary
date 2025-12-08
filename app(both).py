import streamlit as st
import torch
import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re
import sys
from pathlib import Path

# --- SETUP PATHS ---
# Ensure we can import from local files in the current directory
sys.path.append(str(Path(__file__).parent))

# Import Model Architectures
# 1. Regression Model (Student)
from student import StudentModel, load_glove_embeddings
# 2. Contrastive Model
from modules.models import ContrastiveModel
from transformers import AutoTokenizer

# --- CONFIGURATION ---
PAGE_TITLE = "Reverse Dictionary 🧠"
PAGE_ICON = "🔍"

# File Paths
GLOVE_PATH = "glove.6B.300d.txt"
DATASET_PATH = "dataset.json"
CKPT_STUDENT = "ckpts/student_model.pt"
CKPT_CONTRASTIVE = "ckpts/contrastive_model.pt"

# Model Configs
BERT_MODEL_REGRESSION = "bert-base-uncased"       # Used by StudentModel
BERT_MODEL_CONTRASTIVE = "distilbert-base-uncased" # Used by ContrastiveModel
DEVICE = "cpu"  # Safer for local demos

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- RESOURCE LOADING (Cached) ---
@st.cache_resource
def load_data_and_glove():
    """Loads GloVe and the Dataset (Shared by both models)"""
    # 1. Load GloVe
    glove = load_glove_embeddings(GLOVE_PATH)
    dim = next(iter(glove.values())).shape[0]
    
    # 2. Load Dataset & Build Search Index
    with open(DATASET_PATH, 'r', encoding="utf-8") as f:
        items = json.load(f)
    
    db_words = []
    db_vectors = []
    db_definitions = []
    seen_words = set()

    for item in items:
        word = item.get('word')
        defn = item.get('definition', '')
        
        if not word or word in seen_words: 
            continue
        
        # Create Ground Truth Vector (GloVe average)
        # This mirrors the logic in your dataset.py
        tokens = [t for t in re.split(r"[\s_]+", word.lower()) if t]
        vec = None
        if tokens:
            token_vecs = [glove.get(t) for t in tokens if glove.get(t) is not None]
            if token_vecs:
                vec = np.mean(token_vecs, axis=0)
        
        if vec is not None:
            db_words.append(word)
            db_definitions.append(defn)
            db_vectors.append(vec)
            seen_words.add(word)

    db_vectors = np.stack(db_vectors)
    
    # Create torch tensor of full GloVe matrix for Contrastive Model init
    glove_matrix_np = np.array(list(glove.values()))
    glove_matrix_torch = torch.from_numpy(glove_matrix_np)
    
    return dim, db_words, db_vectors, db_definitions, glove_matrix_torch

@st.cache_resource
def load_models(dim, _glove_matrix_torch):
    """Loads both Neural Networks and their Tokenizers"""
    
    # --- 1. Load Regression Student ---
    student_model = StudentModel(bert_model_name=BERT_MODEL_REGRESSION, target_dim=dim)
    try:
        # Student model usually saves just the state dict directly
        student_model.load_state_dict(torch.load(CKPT_STUDENT, map_location=torch.device(DEVICE)))
    except FileNotFoundError:
        pass # Handle warning in UI
    except Exception as e:
        st.warning(f"Regression Model Load Error: {e}")
        
    student_model.to(DEVICE)
    student_model.eval()

    # --- 2. Load Contrastive Model ---
    # We init with the FULL GloVe matrix (400k words)
    contrastive_model = ContrastiveModel(
        glove_matrix=_glove_matrix_torch,
        text_model_name=BERT_MODEL_CONTRASTIVE, 
        image_backbone="resnet50",
        freeze_text=False,
        freeze_image=False,
        freeze_word_embeddings=True
    )
    
    try:
        # Load the raw checkpoint
        checkpoint = torch.load(CKPT_CONTRASTIVE, map_location=torch.device(DEVICE))
        
        # Extract the state dict if it's wrapped in metadata (epoch, etc.)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # === FIX FOR SIZE MISMATCH (997 vs 400k) ===
        # The checkpoint has a 'word_emb.weight' of size [997, 300].
        # Our model has 'word_emb.weight' of size [400000, 300].
        # Since we only use the Text Encoder for inference (and not the word embeddings),
        # we can safely delete this key from the checkpoint to avoid the size mismatch error.
        if 'word_emb.weight' in state_dict:
            del state_dict['word_emb.weight']
            
        # Load with strict=False (allows missing word_emb.weight)
        contrastive_model.load_state_dict(state_dict, strict=False)
        
    except FileNotFoundError:
        pass # Handle warning in UI
    except Exception as e:
        st.warning(f"Contrastive Model Load Warning: {e}")

    contrastive_model.to(DEVICE)
    contrastive_model.eval()

    # --- Tokenizers ---
    tokenizer_student = AutoTokenizer.from_pretrained(BERT_MODEL_REGRESSION)
    tokenizer_contrastive = AutoTokenizer.from_pretrained(BERT_MODEL_CONTRASTIVE)

    return student_model, contrastive_model, tokenizer_student, tokenizer_contrastive

# --- MAIN APP LOGIC ---

st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("### The \"Tip of the Tongue\" Solver")

# 1. Load Data
with st.spinner('Loading Embeddings & Search Index... (First run takes ~45s)'):
    dim, db_words, db_vectors, db_definitions, glove_matrix_torch = load_data_and_glove()

# 2. Load Models
with st.spinner('Loading Neural Networks...'):
    student_model, contrastive_model, tokenizer_student, tokenizer_contrastive = load_models(dim, glove_matrix_torch)

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.radio(
        "Choose Model Architecture:",
        ["Regression (Student)", "Contrastive (CLIP-Style)"]
    )
    
    k_val = st.slider("Top K Matches", 1, 20, 5)
    st.divider()
    st.info(f"""
    **Current Architecture:** {model_choice}
    \n**Vocabulary Size:** {len(db_words):,} words
    """)

# 4. Search UI
query = st.text_input("Describe the concept:", placeholder="A large boat that carries planes...")
search_btn = st.button("Find Word 🚀", type="primary")

if search_btn and query:
    with st.spinner('Thinking...'):
        predicted_vector = None
        
        # --- PATH A: REGRESSION MODEL ---
        if model_choice == "Regression (Student)":
            if not student_model:
                st.error("Regression Model checkpoint missing.")
                st.stop()
                
            inputs = tokenizer_student(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                predicted_vector = student_model(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE))
            pred_vec_np = predicted_vector.cpu().numpy()

        # --- PATH B: CONTRASTIVE MODEL ---
        elif model_choice == "Contrastive (CLIP-Style)":
            if not contrastive_model:
                st.error("Contrastive Model checkpoint missing.")
                st.stop()
                
            inputs = tokenizer_contrastive(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                # Call .encode_text() specifically for the contrastive architecture
                predicted_vector = contrastive_model.encode_text(inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE))
            pred_vec_np = predicted_vector.cpu().numpy()

        # --- SEARCH & DISPLAY ---
        
        # 1. Normalize Predicted Vector (Cosine Similarity requires normalization)
        norm = np.linalg.norm(pred_vec_np)
        if norm > 0:
            pred_vec_np = pred_vec_np / norm
            
        # 2. Cosine Similarity against Database
        # sklearn's cosine_similarity handles the broadcasting automatically
        sims = cosine_similarity(pred_vec_np, db_vectors).flatten()

        # 3. Sort Results
        top_k_indices = sims.argsort()[-k_val:][::-1]

        # 4. Format Output
        results = []
        for idx in top_k_indices:
            results.append({
                "Word": db_words[idx],
                "Confidence": float(sims[idx]),
                "Definition": db_definitions[idx]
            })
        
        st.success(f"Search Complete using **{model_choice}**!")
        
        # Show Top Match
        top_match = results[0]
        st.markdown(f"## 🏆 Top Match: **:blue[{top_match['Word']}]**")
        st.caption(f"Dictionary Def: {top_match['Definition']}")
        
        # Plot Graph
        df = pd.DataFrame(results)
        fig = px.bar(
            df, x='Confidence', y='Word', orientation='h', 
            color='Confidence', title="Semantic Similarity Scores",
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Show Table
        with st.expander("View Raw Data"):
            st.dataframe(df)