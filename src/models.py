"""
models.py

Hierarchical binary logistic regression models for the
patient–physician semantic alignment study.

This module is limited to:
1. constructing the four manuscript model formulas;
2. preparing a common complete-case analysis sample; and
3. fitting the logistic regression models.

Author: Isaac Owusu Asante
"""

from collections import OrderedDict
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# Default variable names
# ---------------------------------------------------------------------

DEFAULT_TARGET = "lower_rating"

DEFAULT_SEMANTIC_ALIGNMENT = "alignment_semantic_z"

DEFAULT_TFIDF_ALIGNMENT = "alignment_tfidf_z"

DEFAULT_PLATFORM_CONTROLS = [
    "log_review_count_z",
    "log_patient_count_z",
    "patient_need_length_main_z",
]

DEFAULT_CONSULTATION_TYPE = "评论类型"

DEFAULT_PROVINCE = "province"


# ---------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------

def quote_variable(variable_name: str) -> str:
    """
    Safely quote a variable name for a Patsy formula.

    This supports Chinese variable names, spaces, punctuation,
    and other non-standard column names.
    """

    escaped = variable_name.replace('"', '\\"')

    return f'Q("{escaped}")'


def categorical_term(variable_name: str) -> str:
    """
    Return a categorical Patsy formula term.
    """

    return f"C({quote_variable(variable_name)})"


# ---------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------

def validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Confirm that all required model variables exist.
    """

    required_columns = list(dict.fromkeys(required_columns))

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing variables required for model estimation: "
            f"{missing}"
        )


# ---------------------------------------------------------------------
# Analysis-sample preparation
# ---------------------------------------------------------------------

def prepare_model_data(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    target: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """
    Create the common complete-case sample used by all four models.

    Using the same observations across models ensures that changes
    in model fit are not caused by changes in sample size.
    """

    required_columns = list(dict.fromkeys(required_columns))

    validate_columns(
        df=df,
        required_columns=required_columns,
    )

    model_df = df.loc[:, required_columns].copy()

    numeric_columns = model_df.select_dtypes(
        include=[np.number]
    ).columns

    model_df.loc[:, numeric_columns] = (
        model_df.loc[:, numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    model_df = model_df.dropna().copy()

    if model_df.empty:
        raise ValueError(
            "No observations remain after complete-case filtering."
        )

    target_values = set(
        model_df[target].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Outcome variable '{target}' must contain only 0 and 1."
        )

    if len(target_values) != 2:
        raise ValueError(
            f"Outcome variable '{target}' must contain both classes."
        )

    return model_df


# ---------------------------------------------------------------------
# Formula construction
# ---------------------------------------------------------------------

def build_model_formulas(
    target: str = DEFAULT_TARGET,
    semantic_alignment: str = DEFAULT_SEMANTIC_ALIGNMENT,
    tfidf_alignment: str = DEFAULT_TFIDF_ALIGNMENT,
    platform_controls: Optional[Iterable[str]] = None,
    consultation_type: str = DEFAULT_CONSULTATION_TYPE,
    province: str = DEFAULT_PROVINCE,
) -> OrderedDict:
    """
    Construct the four hierarchical logistic regression formulas.

    Model 1
        Semantic alignment only.

    Model 2
        Semantic alignment plus platform-related controls.

    Model 3
        Model 2 plus consultation type and province fixed effects.

    Model 4
        Model 3 plus TF-IDF lexical alignment.
    """

    if platform_controls is None:
        platform_controls = DEFAULT_PLATFORM_CONTROLS

    platform_controls = list(platform_controls)

    outcome = quote_variable(target)
    semantic = quote_variable(semantic_alignment)
    tfidf = quote_variable(tfidf_alignment)

    control_terms = [
        quote_variable(column)
        for column in platform_controls
    ]

    consultation = categorical_term(
        consultation_type
    )

    province_fixed_effects = categorical_term(
        province
    )

    model_1_terms = [
        semantic,
    ]

    model_2_terms = [
        semantic,
        *control_terms,
    ]

    model_3_terms = [
        semantic,
        *control_terms,
        consultation,
        province_fixed_effects,
    ]

    model_4_terms = [
        semantic,
        tfidf,
        *control_terms,
        consultation,
        province_fixed_effects,
    ]

    formulas = OrderedDict()

    formulas["Model 1"] = (
        f"{outcome} ~ "
        + " + ".join(model_1_terms)
    )

    formulas["Model 2"] = (
        f"{outcome} ~ "
        + " + ".join(model_2_terms)
    )

    formulas["Model 3"] = (
        f"{outcome} ~ "
        + " + ".join(model_3_terms)
    )

    formulas["Model 4"] = (
        f"{outcome} ~ "
        + " + ".join(model_4_terms)
    )

    return formulas


# ---------------------------------------------------------------------
# Individual model fitting
# ---------------------------------------------------------------------

def fit_logistic_model(
    formula: str,
    data: pd.DataFrame,
    maxiter: int = 200,
    disp: bool = False,
):
    """
    Fit one binary logistic regression model.
    """

    model = smf.logit(
        formula=formula,
        data=data,
        missing="drop",
    )

    result = model.fit(
        method="lbfgs",
        maxiter=500,
        disp=disp,
    )

    return result


# ---------------------------------------------------------------------
# Full hierarchical model fitting
# ---------------------------------------------------------------------

def fit_hierarchical_models(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    semantic_alignment: str = DEFAULT_SEMANTIC_ALIGNMENT,
    tfidf_alignment: str = DEFAULT_TFIDF_ALIGNMENT,
    platform_controls: Optional[Iterable[str]] = None,
    consultation_type: str = DEFAULT_CONSULTATION_TYPE,
    province: str = DEFAULT_PROVINCE,
    maxiter: int = 200,
    disp: bool = False,
):
    """
    Fit all four logistic regression models reported in the manuscript.

    Returns
    -------
    fitted_models
        Ordered dictionary containing the four fitted model objects.

    model_data
        Common complete-case DataFrame used for all four models.

    formulas
        Ordered dictionary containing the four model formulas.
    """

    if platform_controls is None:
        platform_controls = DEFAULT_PLATFORM_CONTROLS

    platform_controls = list(platform_controls)

    required_columns = [
        target,
        semantic_alignment,
        tfidf_alignment,
        *platform_controls,
        consultation_type,
        province,
    ]

    model_data = prepare_model_data(
        df=df,
        required_columns=required_columns,
        target=target,
    )

    formulas = build_model_formulas(
        target=target,
        semantic_alignment=semantic_alignment,
        tfidf_alignment=tfidf_alignment,
        platform_controls=platform_controls,
        consultation_type=consultation_type,
        province=province,
    )

    fitted_models = OrderedDict()

    for model_name, formula in formulas.items():

        fitted_models[model_name] = fit_logistic_model(
            formula=formula,
            data=model_data,
            maxiter=maxiter,
            disp=disp,
        )

    return fitted_models, model_data, formulas