"""
semantic_alignment.py

Semantic alignment between patient needs and physician expertise
using Sentence-BERT embeddings.

This module computes semantic similarity scores that are later
used as predictors in the regression and machine-learning models.

Author: Isaac Owusu Asante
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------------------

def load_embedding_model(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
):
    """
    Load Sentence-BERT model.

    Returns
    -------
    SentenceTransformer
    """

    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------
# Encode text
# ---------------------------------------------------------------------

def encode_text(
    model,
    texts,
    batch_size=64,
    show_progress=True,
):
    """
    Convert text into embeddings.
    """

    embeddings = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


# ---------------------------------------------------------------------
# Semantic alignment
# ---------------------------------------------------------------------

def semantic_alignment(
    patient_embeddings,
    physician_embeddings,
):
    """
    Cosine similarity between patient
    and physician embeddings.
    """

    similarity = cosine_similarity(
        patient_embeddings,
        physician_embeddings,
    )

    return similarity.diagonal()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def compute_semantic_alignment(
    df,
    patient_column="patient_need_text_main",
    physician_column="physician_expertise_text_main",
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    Compute semantic alignment.

    Adds:
        alignment_semantic
    """

    model = load_embedding_model(model_name)

    patient_embeddings = encode_text(
        model,
        df[patient_column],
    )

    physician_embeddings = encode_text(
        model,
        df[physician_column],
    )

    df["alignment_semantic"] = semantic_alignment(
        patient_embeddings,
        physician_embeddings,
    )

    return df


# ---------------------------------------------------------------------
# Optional cache
# ---------------------------------------------------------------------

def save_embeddings(
    embeddings,
    filename,
):
    """
    Save embeddings to disk.
    """

    Path(filename).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(filename, embeddings)


def load_embeddings(filename):
    """
    Load cached embeddings.
    """

    return np.load(filename)