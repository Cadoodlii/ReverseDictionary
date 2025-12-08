import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# --- SETUP PATHS ---
# Ensure we can import from local files
sys.path.append(str(Path(__file__).parent))

# Import your own code
from modules.datasets import MultimodalRD  # (not strictly needed here, but fine)
from modules.models import ContrastiveModel, TextModel
from modules.utils import load_glove_embeddings  # <-- use your own loader

# --- CONFIGURATION ---
PAGE_TITLE = "Reverse Dictionary 🧠"
PAGE_ICON = "🔍"
GLOVE_PATH = "glove.6B.300d.txt"
DATASET_PATH = "dataset.json"
BERT_MODEL = "distilbert-base-uncased"
DEVICE = "mps"  # or "cuda" / "cpu"

# Checkpoint Paths
CKPT_STUDENT = "ckpts/text_model.pt"  # for TextModel
CKPT_CONTRASTIVE = "ckpts/contrastive_model.pt"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")


# --- CACHED RESOURCES ---
@st.cache_resource
def load_data_and_glove():
  """Loads GloVe and the Dataset (Shared by both models)."""
  # 1. Load GloVe
  glove = load_glove_embeddings(GLOVE_PATH)
  dim = next(iter(glove.values())).shape[0]

  # 2. Load Dataset & Build Search Index
  with open(DATASET_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

  db_words = []
  db_vectors = []
  db_definitions = []
  seen_words = set()

  for item in items:
    word = item.get("word")
    defn = item.get("definition", "")

    if not word or word in seen_words:
      continue

    # Create "ground truth" vector: GloVe average over word tokens
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

  # Build full GloVe matrix for initializing models (TextModel/ContrastiveModel)
  glove_vocab = list(glove.keys())
  glove_matrix_np = np.array(list(glove.values()))
  glove_matrix_torch = torch.from_numpy(glove_matrix_np)

  return (
    glove,
    dim,
    db_words,
    db_vectors,
    db_definitions,
    glove_matrix_torch,
  )


from pathlib import Path


@st.cache_resource
def load_models(dim, _glove_matrix_torch_unused):
  """Loads both Neural Networks (TextModel and ContrastiveModel)."""

  device = torch.device(DEVICE)

  # --- Figure out the vocab size & emb dim from any available checkpoint ---
  vocab_size = None
  emb_dim = None

  # Try student checkpoint first
  if Path(CKPT_STUDENT).exists():
    state = torch.load(CKPT_STUDENT, map_location="cpu")
    we = state["model_state_dict"]["word_emb.weight"]
    vocab_size, emb_dim = we.shape
  # Otherwise try contrastive checkpoint
  elif Path(CKPT_CONTRASTIVE).exists():
    state = torch.load(CKPT_CONTRASTIVE, map_location="cpu")
    we = state["model_state_dict"]["word_emb.weight"]
    vocab_size, emb_dim = we.shape
  else:
    # No checkpoints at all: fall back to something reasonable
    # e.g. use dim from GloVe and a small dummy vocab
    vocab_size = 1000
    emb_dim = dim

  dummy_glove_matrix = torch.zeros((vocab_size, emb_dim), dtype=torch.float32)

  # --- 1. Text-only model (Regression-style student) ---
  student_model = TextModel(
    glove_matrix=dummy_glove_matrix,
    text_model_name=BERT_MODEL,
  ).to(device)
  student_ckpt_loaded = False
  try:
    state = torch.load(CKPT_STUDENT, map_location=device)
    student_model.load_state_dict(state["model_state_dict"], strict=True)
    student_ckpt_loaded = True
  except FileNotFoundError:
    st.warning(
      f"⚠️ Could not find student checkpoint at {CKPT_STUDENT}. "
      "The regression model will use random weights."
    )
  except Exception as e:
    st.warning(f"Student model load warning: {e}")
  student_model.eval()

  # --- 2. Contrastive model ---
  contrastive_model = ContrastiveModel(
    glove_matrix=dummy_glove_matrix,
    text_model_name=BERT_MODEL,
    image_backbone="resnet50",
    freeze_text=False,
    freeze_image=False,
    freeze_word_embeddings=True,
  ).to(device)

  contrastive_ckpt_loaded = False
  try:
    state = torch.load(CKPT_CONTRASTIVE, map_location=device)
    contrastive_model.load_state_dict(state["model_state_dict"], strict=True)
    contrastive_ckpt_loaded = True
  except FileNotFoundError:
    st.warning(
      f"⚠️ Could not find contrastive checkpoint at {CKPT_CONTRASTIVE}. "
      "The contrastive model will use random weights."
    )
  except Exception as e:
    st.warning(f"Contrastive model load warning: {e}")
  contrastive_model.eval()

  # --- Tokenizers (same text model for both) ---
  tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

  return (
    student_model,
    contrastive_model,
    tokenizer,
    tokenizer,  # for now both use the same tokenizer
    student_ckpt_loaded,
    contrastive_ckpt_loaded,
  )


# --- MAIN APP ---

st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown('### The "Tip of the Tongue" Solver')

# Load Data (Spinner)
with st.spinner("Loading Embeddings & Search Index... (First run can take a while)"):
  (
    glove,
    dim,
    db_words,
    db_vectors,
    db_definitions,
    glove_matrix_torch,
  ) = load_data_and_glove()

# Load Models (Spinner)
with st.spinner("Loading Neural Networks..."):
  (
    student_model,
    contrastive_model,
    tokenizer_student,
    tokenizer_contrastive,
    student_ckpt_loaded,
    contrastive_ckpt_loaded,
  ) = load_models(dim, glove_matrix_torch)

device = torch.device(DEVICE)

# Sidebar Configuration
with st.sidebar:
  st.header("⚙️ Configuration")
  model_choice = st.radio(
    "Choose Model Architecture:",
    ["Regression (Student)", "Contrastive (CLIP-Style)"],
  )

  k_val = st.slider("Top K Matches", 1, 20, 5)
  st.divider()
  st.info(
    """
    **Regression Model:** Text-only `TextModel` trained to predict GloVe-like vectors.\n
    **Contrastive Model:** `ContrastiveModel` aligning text/image in a shared space (InfoNCE).
    """
  )

# Main Search UI
query = st.text_input(
  "Describe the concept:", placeholder="A large boat that carries planes..."
)
search_btn = st.button("Find Word 🚀", type="primary")

if search_btn and query:
  with st.spinner("Thinking..."):
    predicted_vector = None

    # --- PATH A: REGRESSION (TextModel) ---
    if model_choice == "Regression (Student)":
      if not student_ckpt_loaded:
        st.warning(
          "Student checkpoint not loaded. Results will be based on randomly initialized weights."
        )
      inputs = tokenizer_student(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
      )
      with torch.no_grad():
        predicted_vector = student_model.encode_text(
          inputs["input_ids"].to(device),
          inputs["attention_mask"].to(device),
        )
      pred_vec_np = predicted_vector.cpu().numpy()

    # --- PATH B: CONTRASTIVE (text branch only) ---
    elif model_choice == "Contrastive (CLIP-Style)":
      if not contrastive_ckpt_loaded:
        st.warning(
          "Contrastive checkpoint not loaded. Results will be based on randomly initialized weights."
        )
      inputs = tokenizer_contrastive(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
      )
      with torch.no_grad():
        predicted_vector = contrastive_model.encode_text(
          inputs["input_ids"].to(device),
          inputs["attention_mask"].to(device),
        )
      pred_vec_np = predicted_vector.cpu().numpy()

    # --- SEARCH & DISPLAY ---

    # Normalize predicted vector just in case
    norm = np.linalg.norm(pred_vec_np)
    if norm > 0:
      pred_vec_np = pred_vec_np / norm

    # Cosine similarity against DB vectors (GloVe-based index)
    sims = cosine_similarity(pred_vec_np, db_vectors).flatten()  # [V]

    # Sort by similarity
    top_k_indices = sims.argsort()[-k_val:][::-1]

    # Build result list
    results = []
    for idx in top_k_indices:
      results.append(
        {
          "Word": db_words[idx],
          "Confidence": float(sims[idx]),
          "Definition": db_definitions[idx],
        }
      )

    st.success(f"Search Complete using **{model_choice}**!")

    # Top match
    top_match = results[0]
    st.markdown(f"## 🏆 Top Match: **:blue[{top_match['Word']}]**")
    st.caption(f"Dictionary Def: {top_match['Definition']}")

    # Bar chart
    df = pd.DataFrame(results)
    fig = px.bar(
      df,
      x="Confidence",
      y="Word",
      orientation="h",
      color="Confidence",
      title="Semantic Similarity Scores",
      color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # Raw table
    with st.expander("View Raw Data"):
      st.dataframe(df)
