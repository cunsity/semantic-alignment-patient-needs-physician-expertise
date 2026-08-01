"""
diagnostics.py

Statistical diagnostics and manuscript table generation
for the paper:

Patient–Physician Fit in Online Healthcare Platforms:
Measuring Symptom–Specialty Alignment with Large
Language Model Embeddings.

This module provides:

• descriptive statistics
• correlation analysis
• multicollinearity diagnostics
• logistic regression summaries
• model fit statistics
• manuscript-ready tables
• Excel export utilities

Author:
Isaac Owusu Asante
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)

###############################################################################
# Helper functions
###############################################################################

def significance_stars(p):

    """
    Convert p-values into significance stars.
    """

    if p < 0.001:
        return "***"

    if p < 0.01:
        return "**"

    if p < 0.05:
        return "*"

    return ""


def ensure_directory(directory):

    """
    Create output directory if needed.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


###############################################################################
# Descriptive statistics
###############################################################################

def descriptive_statistics(
    df,
    variables,
    decimals=3,
):

    """
    Produce manuscript descriptive statistics.
    """

    table = (
        df[variables]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
        .round(decimals)
    )

    table.columns = [
        "Mean",
        "SD",
        "Minimum",
        "Q1",
        "Median",
        "Q3",
        "Maximum",
    ]

    return table


###############################################################################
# Correlation matrix
###############################################################################

def correlation_matrix(
    df,
    variables,
    decimals=3,
):

    """
    Compute Pearson correlations
    together with p-values.
    """

    correlation = pd.DataFrame(
        index=variables,
        columns=variables,
        dtype=float,
    )

    pvalues = pd.DataFrame(
        index=variables,
        columns=variables,
        dtype=float,
    )

    for variable1 in variables:

        for variable2 in variables:

            if variable1 == variable2:

               correlation.loc[
                    variable1,
                    variable2,
               ] = 1.0

               pvalues.loc[
                    variable1,
                    variable2,
               ] = 0.0
               continue    
            subset = (
                df.loc[
                    :,
                    [variable1, variable2],
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .dropna()
            )

            r, p = pearsonr(
                subset[variable1].to_numpy(),
                subset[variable2].to_numpy(),
            )

    return (
        correlation.round(decimals),
        pvalues,
    )


def manuscript_correlation_table(
    correlation,
    pvalues,
    decimals=3,
):

    """
    Add significance stars
    to correlation coefficients.
    """

    table = correlation.copy().astype(str)

    for row in correlation.index:

        for col in correlation.columns:

            value = correlation.loc[row, col]

            stars = significance_stars(
                pvalues.loc[row, col]
            )

            table.loc[row, col] = (
                f"{value:.{decimals}f}{stars}"
            )

    return table
###############################################################################
# Variance Inflation Factors (VIF)
###############################################################################

def compute_vif(
    df,
    variables,
    decimals=3,
):
    """
    Compute Variance Inflation Factors (VIF) and tolerance.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset.

    variables : list
        Continuous predictor variables.

    Returns
    -------
    pandas.DataFrame
        VIF table.
    """

    x = (
        df[variables]
        .dropna()
        .copy()
    )

    rows = []

    for index, variable in enumerate(variables):

        vif = variance_inflation_factor(
            x.values,
            index,
        )

        tolerance = (
            np.nan
            if np.isinf(vif)
            else 1 / vif
        )

        rows.append(
            {
                "Variable": variable,
                "VIF": round(vif, decimals),
                "Tolerance": round(
                    tolerance,
                    decimals,
                ),
            }
        )

    vif_table = pd.DataFrame(rows)

    return vif_table


###############################################################################
# Logistic regression extraction
###############################################################################

def extract_coefficients(
    fitted_model,
    decimals=3,
):
    """
    Extract regression coefficients.

    Returns
    -------
    pandas.DataFrame
    """

    params = fitted_model.params

    std_errors = fitted_model.bse

    z_values = fitted_model.tvalues

    p_values = fitted_model.pvalues

    results = pd.DataFrame(
        {
            "Coefficient": params,
            "Std. Error": std_errors,
            "z": z_values,
            "P": p_values,
        }
    )

    results["Stars"] = (
        results["P"]
        .apply(significance_stars)
    )

    return results.round(decimals)


def extract_odds_ratios(
    fitted_model,
    decimals=3,
):
    """
    Compute Odds Ratios
    together with 95% confidence intervals.
    """

    coefficients = fitted_model.params

    confidence = fitted_model.conf_int()

    odds_ratio = np.exp(coefficients)

    lower = np.exp(
        confidence.iloc[:, 0]
    )

    upper = np.exp(
        confidence.iloc[:, 1]
    )

    results = pd.DataFrame(
        {
            "Odds Ratio": odds_ratio,
            "Lower CI": lower,
            "Upper CI": upper,
            "P": fitted_model.pvalues,
        }
    )

    results["Stars"] = (
        results["P"]
        .apply(significance_stars)
    )

    return results.round(decimals)


###############################################################################
# Multiple-model extraction
###############################################################################

def extract_all_models(
    fitted_models,
):
    """
    Extract coefficients and odds ratios
    for every fitted model.

    Parameters
    ----------
    fitted_models : OrderedDict

    Returns
    -------
    dict
    """

    output = {}

    for model_name, model in fitted_models.items():

        output[model_name] = {
            "coefficients":
                extract_coefficients(model),

            "odds_ratios":
                extract_odds_ratios(model),
        }

    return output
###############################################################################
# Model fit statistics
###############################################################################

def model_fit_statistics(
    fitted_model,
    decimals=3,
):
    """
    Extract overall model fit statistics.

    Parameters
    ----------
    fitted_model : statsmodels LogitResults

    Returns
    -------
    dict
    """

    return {

        "Observations":
            int(fitted_model.nobs),

        "Log Likelihood":
            round(fitted_model.llf, decimals),

        "Null Log Likelihood":
            round(fitted_model.llnull, decimals),

        "Likelihood Ratio":
            round(fitted_model.llr, decimals),

        "LR p-value":
            round(fitted_model.llr_pvalue, decimals),

        "AIC":
            round(fitted_model.aic, decimals),

        "BIC":
            round(fitted_model.bic, decimals),

        "McFadden R²":
            round(fitted_model.prsquared, decimals),

        "Converged":
            fitted_model.mle_retvals.get(
                "converged",
                np.nan,
            ),
    }


def extract_model_fit(
    fitted_models,
):
    """
    Extract fit statistics for every model.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for model_name, model in fitted_models.items():

        row = model_fit_statistics(model)

        row["Model"] = model_name

        rows.append(row)

    columns = [
        "Model",
        "Observations",
        "Log Likelihood",
        "Null Log Likelihood",
        "Likelihood Ratio",
        "LR p-value",
        "AIC",
        "BIC",
        "McFadden R²",
        "Converged",
    ]

    return pd.DataFrame(rows)[columns]


###############################################################################
# Publication-ready regression table
###############################################################################

def manuscript_regression_table(
    fitted_models,
    decimals=3,
):
    """
    Produce a side-by-side regression table
    similar to the manuscript.

    Each cell reports

    Odds Ratio*** [Lower CI, Upper CI]
    """

    variables = set()

    for model in fitted_models.values():

        variables.update(
            model.params.index
        )

    variables = list(variables)

    table = pd.DataFrame(
        index=variables
    )

    for model_name, model in fitted_models.items():

        odds = np.exp(
            model.params
        )

        conf = np.exp(
            model.conf_int()
        )

        pvals = model.pvalues

        formatted = []

        for variable in variables:

            if variable not in odds.index:

                formatted.append("")

                continue

            stars = significance_stars(
                pvals[variable]
            )

            value = (
                f"{odds[variable]:.{decimals}f}"
                f"{stars}"
                f" [{conf.loc[variable,0]:.{decimals}f}, "
                f"{conf.loc[variable,1]:.{decimals}f}]"
            )

            formatted.append(value)

        table[model_name] = formatted

    return table


###############################################################################
# Compact model comparison
###############################################################################

def model_comparison_table(
    fitted_models,
    decimals=3,
):
    """
    Produce a compact comparison
    of the four hierarchical models.
    """

    rows = []

    for model_name, model in fitted_models.items():

        rows.append(
            {
                "Model":
                    model_name,

                "Observations":
                    int(model.nobs),

                "Pseudo R²":
                    round(
                        model.prsquared,
                        decimals,
                    ),

                "AIC":
                    round(
                        model.aic,
                        decimals,
                    ),

                "BIC":
                    round(
                        model.bic,
                        decimals,
                    ),

                "LL":
                    round(
                        model.llf,
                        decimals,
                    ),
            }
        )

    return pd.DataFrame(rows)
###############################################################################
# Excel export utilities
###############################################################################

def export_tables(
    tables: Dict[str, pd.DataFrame],
    output_directory="results/tables",
):
    """
    Export all manuscript tables to Excel.

    Parameters
    ----------
    tables : dict
        Dictionary whose values are pandas DataFrames.

    output_directory : str
        Output folder.
    """

    output_directory = ensure_directory(
        output_directory
    )

    exported_files = {}

    for name, table in tables.items():

        filename = (
            output_directory /
            f"{name}.xlsx"
        )

        table.to_excel(
            filename,
            index=True,
        )

        exported_files[name] = filename

    return exported_files


###############################################################################
# Master diagnostics function
###############################################################################

def run_diagnostics(
    df,
    fitted_models,
    descriptive_variables,
    correlation_variables,
    vif_variables,
    output_directory="results/tables",
):
    """
    Run the complete diagnostics pipeline.

    Parameters
    ----------
    df : pandas.DataFrame

    fitted_models : OrderedDict

    descriptive_variables : list

    correlation_variables : list

    vif_variables : list

    output_directory : str

    Returns
    -------
    dict
        Dictionary containing every manuscript table.
    """

    # ----------------------------------------------------------
    # Descriptive statistics
    # ----------------------------------------------------------

    descriptives = descriptive_statistics(
        df,
        descriptive_variables,
    )

    # ----------------------------------------------------------
    # Correlations
    # ----------------------------------------------------------

    corr, pvals = correlation_matrix(
        df,
        correlation_variables,
    )

    corr_table = manuscript_correlation_table(
        corr,
        pvals,
    )

    # ----------------------------------------------------------
    # Multicollinearity
    # ----------------------------------------------------------

    vif = compute_vif(
        df,
        vif_variables,
    )

    # ----------------------------------------------------------
    # Regression tables
    # ----------------------------------------------------------

    regression = manuscript_regression_table(
        fitted_models,
    )

    # ----------------------------------------------------------
    # Model comparison
    # ----------------------------------------------------------

    comparison = model_comparison_table(
        fitted_models,
    )

    fit = extract_model_fit(
        fitted_models,
    )

    tables = {

        "table1_descriptive_statistics":
            descriptives,

        "table2_correlations":
            corr_table,

        "table3_vif":
            vif,

        "table4_logistic_regression":
            regression,

        "table5_model_comparison":
            comparison,

        "model_fit":
            fit,
    }

    exported = export_tables(
        tables,
        output_directory,
    )

    return {

        "tables": tables,

        "exported_files": exported,

    }


###############################################################################
# Convenience printing
###############################################################################

def print_model_summary(
    fitted_models,
):
    """
    Print Statsmodels summaries
    for all fitted models.
    """

    for model_name, model in fitted_models.items():

        print()

        print("=" * 80)

        print(model_name)

        print("=" * 80)

        print()

        print(model.summary())


###############################################################################
# End of module
###############################################################################
