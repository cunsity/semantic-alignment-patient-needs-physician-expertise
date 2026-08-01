"""
features.py

Feature engineering for the Semantic Patient–Physician Alignment study.

This module creates all engineered variables used in the
statistical and machine-learning analyses.

Author: Isaac Owusu Asante
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Outcome variable
# ---------------------------------------------------------------------

def create_lower_rating(df):
    """
    Binary outcome variable.

    1 = rating < 5
    0 = rating = 5
    """

    df["lower_rating"] = (
        df["评论打分"]
        .astype(float)
        .lt(5)
        .astype(int)
    )

    return df


# ---------------------------------------------------------------------
# Log transformations
# ---------------------------------------------------------------------

def create_log_variables(df):
    """
    Log-transform skewed count variables.
    """

    count_vars = {
        "评价数量": "log_review_count",
        "患者数量": "log_patient_count",
    }

    for original, new in count_vars.items():

        if original in df.columns:

            df[new] = np.log1p(
                pd.to_numeric(
                    df[original],
                    errors="coerce"
                ).fillna(0)
            )

    return df


# ---------------------------------------------------------------------
# TF-IDF alignment
# ---------------------------------------------------------------------

def compute_tfidf_alignment(df):
    """
    Lexical alignment between
    patient needs and physician expertise.
    """

    corpus = pd.concat(
        [
            df["patient_need_text_main"],
            df["physician_expertise_text_main"],
        ],
        ignore_index=True,
    )

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
    )

    vectorizer.fit(corpus)

    patient_matrix = vectorizer.transform(
        df["patient_need_text_main"]
    )

    physician_matrix = vectorizer.transform(
        df["physician_expertise_text_main"]
    )

    similarity = cosine_similarity(
        patient_matrix,
        physician_matrix,
    )

    df["alignment_tfidf"] = similarity.diagonal()

    return df


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def standardize_features(df):
    """
    Create z-score versions of
    continuous predictors.
    """

    scaler = StandardScaler()

    continuous = []

    candidates = [

        "alignment_tfidf",

        "patient_need_length_main",

        "physician_expertise_length_main",

        "log_review_count",

        "log_patient_count",

    ]

    for c in candidates:

        if c in df.columns:

            continuous.append(c)

    if len(continuous) > 0:

        df[
            [f"{c}_z" for c in continuous]
        ] = scaler.fit_transform(
            df[continuous]
        )

    return df


# ---------------------------------------------------------------------
# Physician title
# ---------------------------------------------------------------------

def simplify_physician_title(df):
    """
    Simplify physician titles.

    Optional.
    """

    if "医生职称" not in df.columns:

        return df

    df["physician_title"] = (
        df["医生职称"]
        .fillna("Unknown")
        .astype(str)
    )

    return df


# ---------------------------------------------------------------------
# Province
# ---------------------------------------------------------------------

def clean_province(df):

    if "所在省份" not in df.columns:

        return df

    df["province"] = (
        df["所在省份"]
        .fillna("Unknown")
        .astype(str)
    )

    return df


# ---------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------

def engineer_features(df):
    """
    Complete feature engineering pipeline.
    """

    df = create_lower_rating(df)

    df = create_log_variables(df)

    df = compute_tfidf_alignment(df)

    df = standardize_features(df)

    df = simplify_physician_title(df)

    df = clean_province(df)

    return df