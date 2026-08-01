"""
preprocessing.py

Data loading and preprocessing for the
Semantic Patient–Physician Alignment study.

This module reproduces the preprocessing pipeline
used in the manuscript.

Author: Isaac Owusu Asante
"""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "评论打分",
    "评论症状",
    "评论内容",
    "医生擅长",
    "医生简介",
    "部门名称",
    "标注部门名称",
]


def load_data(path, encoding="utf-8-sig"):
    """
    Load modelling dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path, encoding=encoding)


def validate_columns(df):
    """
    Ensure all required variables exist.
    """

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


def build_patient_text(df):
    """
    Construct patient-side text.

    IMPORTANT:
    评论标签 is intentionally excluded
    to avoid satisfaction-label leakage.
    """

    df["patient_need_text_main"] = (
        df[
            [
                "评论症状",
                "评论内容",
            ]
        ]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def build_physician_text(df):
    """
    Construct physician-side text.
    """

    physician_cols = [
        c
        for c in [
            "标注部门名称",
            "部门名称",
            "医生擅长",
            "医生简介",
        ]
        if c in df.columns
    ]

    df["physician_expertise_text_main"] = (
        df[physician_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def compute_text_lengths(df):
    """
    Compute text lengths used
    for sample filtering.
    """

    df["patient_need_length_main"] = (
        df["patient_need_text_main"]
        .str.len()
    )

    df["physician_expertise_length_main"] = (
        df["physician_expertise_text_main"]
        .str.len()
    )

    return df


def filter_analysis_sample(df):
    """
    Apply manuscript inclusion criteria.
    """

    return df.loc[
        (
            df["patient_need_length_main"] >= 10
        )
        &
        (
            df["physician_expertise_length_main"] >= 20
        )
    ].copy()


def preprocess(path):
    """
    Complete preprocessing pipeline.
    """

    df = load_data(path)

    validate_columns(df)

    df = build_patient_text(df)

    df = build_physician_text(df)

    df = compute_text_lengths(df)

    df = filter_analysis_sample(df)

    return df