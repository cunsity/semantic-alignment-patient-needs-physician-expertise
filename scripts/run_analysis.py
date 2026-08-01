"""
run_analysis.py

Main reproduction script for

Patient–Physician Fit in Online Healthcare Platforms:
Measuring Symptom–Specialty Alignment with Large Language Models

This script reproduces all analyses reported in the manuscript.

Workflow

1. Load processed dataset
2. Preprocess data
3. Engineer features
4. Compute semantic alignment
5. Fit hierarchical logistic regressions
6. Produce manuscript tables
7. Run Random Forest robustness analysis
8. Save all outputs

Author:
Isaac Owusu Asante
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd

from src.preprocessing import *

from src.features import *

from src.semantic_alignment import *

from src.models import *

from src.diagnostics import *

from src.robustness import *
###############################################################################
# Repository paths
###############################################################################

DATA = ROOT / "data"

RESULTS = ROOT / "results"

TABLES = RESULTS / "tables"

ROBUSTNESS = RESULTS / "robustness"

FIGURES = RESULTS / "figures"

DATA.mkdir(exist_ok=True)

RESULTS.mkdir(exist_ok=True)

TABLES.mkdir(exist_ok=True)

ROBUSTNESS.mkdir(exist_ok=True)

FIGURES.mkdir(exist_ok=True)
###############################################################################
# Load processed dataset
###############################################################################

DATASET = (
    DATA /
    "paper4_final_regression_dataset.csv"
)

if not DATASET.exists():

    raise FileNotFoundError(
        "Dataset not found.\n"
        "Place the processed dataset at:\n"
        f"{DATASET}"
    )

df = pd.read_csv(DATASET)

print()

print("=" * 80)

print("Dataset loaded")

print(df.shape)

print("=" * 80)
###############################################################################
# Preprocess data
###############################################################################

print("\n" + "=" * 80)
print("Step 1: Preprocessing data")
print("=" * 80)

df = preprocess(DATASET)

print(f"Analytical rows after preprocessing: {len(df):,}")
print(f"Columns after preprocessing: {df.shape[1]}")


###############################################################################
# Engineer outcome, controls, and TF-IDF alignment
###############################################################################

print("\n" + "=" * 80)
print("Step 2: Engineering model variables")
print("=" * 80)

df = engineer_features(df)

print("Created:")
print("- lower_rating")
print("- log_review_count")
print("- log_patient_count")
print("- alignment_tfidf")
print("- standardized continuous controls")
print("- province")


###############################################################################
# Compute embedding-based semantic alignment
###############################################################################

print("\n" + "=" * 80)
print("Step 3: Computing embedding-based semantic alignment")
print("=" * 80)

df = compute_semantic_alignment(
    df=df,
    patient_column="patient_need_text_main",
    physician_column="physician_expertise_text_main",
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)

print("Created: alignment_semantic")


###############################################################################
# Standardize semantic alignment
###############################################################################

semantic_mean = df["alignment_semantic"].mean()
semantic_sd = df["alignment_semantic"].std(ddof=0)

if semantic_sd == 0 or pd.isna(semantic_sd):
    raise ValueError(
        "Semantic alignment cannot be standardized because "
        "its standard deviation is zero or missing."
    )

df["alignment_semantic_z"] = (
    df["alignment_semantic"] - semantic_mean
) / semantic_sd


###############################################################################
# Confirm required standardized variables
###############################################################################

required_model_variables = [
    "lower_rating",
    "alignment_semantic",
    "alignment_semantic_z",
    "alignment_tfidf",
    "alignment_tfidf_z",
    "log_review_count",
    "log_review_count_z",
    "log_patient_count",
    "log_patient_count_z",
    "patient_need_length_main",
    "patient_need_length_main_z",
    "评论类型",
    "province",
]

missing_variables = [
    variable
    for variable in required_model_variables
    if variable not in df.columns
]

if missing_variables:
    raise ValueError(
        "The following required variables were not created:\n"
        f"{missing_variables}"
    )

print("\nRequired model variables verified successfully.")


###############################################################################
# Verify final analytical sample
###############################################################################

print("\n" + "=" * 80)
print("Final analytical sample")
print("=" * 80)

print(f"Observations: {len(df):,}")

outcome_counts = (
    df["lower_rating"]
    .value_counts()
    .sort_index()
)

print("\nOutcome distribution:")
print(outcome_counts)

print("\nOutcome percentages:")
print(
    (
        df["lower_rating"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )
)


###############################################################################
# Save analysis-ready variables only
###############################################################################

analysis_columns = [
    "lower_rating",
    "alignment_semantic",
    "alignment_semantic_z",
    "alignment_tfidf",
    "alignment_tfidf_z",
    "log_review_count",
    "log_review_count_z",
    "log_patient_count",
    "log_patient_count_z",
    "patient_need_length_main",
    "patient_need_length_main_z",
    "评论类型",
    "province",
]

analysis_ready = df[analysis_columns].copy()

analysis_ready_path = RESULTS / "analysis_ready_variables.csv"

analysis_ready.to_csv(
    analysis_ready_path,
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nDe-identified analysis-ready variables saved to:\n"
    f"{analysis_ready_path}"
)
###############################################################################
# Hierarchical Logistic Regression
###############################################################################

print("\n" + "=" * 80)
print("Step 4: Hierarchical Logistic Regression")
print("=" * 80)

platform_controls = [
    "log_review_count_z",
    "log_patient_count_z",
    "patient_need_length_main_z",
]

(
    fitted_models,
    model_data,
    formulas,
) = fit_hierarchical_models(
    df=df,
    target="lower_rating",
    semantic_alignment="alignment_semantic_z",
    tfidf_alignment="alignment_tfidf_z",
    platform_controls=platform_controls,
    consultation_type="评论类型",
    province="province",
)

print()

print("Four hierarchical models successfully estimated.")

print()

for model_name in fitted_models.keys():

    print(model_name)

print()

print("=" * 80)
print("Model sample size")
print("=" * 80)

print(model_data.shape)

print()

###############################################################################
# Produce manuscript tables
###############################################################################

print("\n" + "=" * 80)
print("Step 5: Creating manuscript statistical tables")
print("=" * 80)

correlation_variables = [

    "alignment_semantic_z",

    "alignment_tfidf_z",

    "log_review_count_z",

    "log_patient_count_z",

    "patient_need_length_main_z",

]

diagnostics = run_diagnostics(

    df=model_data,

    fitted_models=fitted_models,

    descriptive_variables=correlation_variables,

    correlation_variables=correlation_variables,

    vif_variables=correlation_variables,

)

print()

print("Generated tables:")

for table_name in diagnostics["tables"]:

    print(f"  • {table_name}")

print()

print("Exported files:")

for name, path in diagnostics["exported_files"].items():

    print(f"{name}")

    print(path)

    print()

print("=" * 80)
print("Hierarchical regression completed.")
print("=" * 80)
###############################################################################
# Random Forest Robustness Analysis
###############################################################################

print("\n" + "=" * 80)
print("Step 6: Random Forest Robustness Analysis")
print("=" * 80)

###############################################################################
# Predictor variables
###############################################################################

rf_predictors = [

    "alignment_semantic_z",

    "alignment_tfidf_z",

    "log_review_count_z",

    "log_patient_count_z",

    "patient_need_length_main_z",

    "province",

    "评论类型",

]

###############################################################################
# Run robustness analysis
###############################################################################

rf_results = run_random_forest_robustness(

    df=df,

    predictors=rf_predictors,

    outcome="lower_rating",

    categorical_variables=[

        "province",

        "评论类型",

    ],

    test_size=0.20,

    random_state=42,

    n_estimators=500,

    cross_validation_folds=5,

    permutation_repeats=10,

    output_directory=ROBUSTNESS,

    export=True,

)

###############################################################################
# Display summary metrics
###############################################################################

print()

print("=" * 80)
print("Random Forest Performance")
print("=" * 80)

metrics = rf_results["metrics"]

for metric, value in metrics.items():

    print(f"{metric:15s}: {value:.4f}")

print()

print(f"ROC-AUC        : {rf_results['roc_auc']:.4f}")

###############################################################################
# Display feature importance
###############################################################################

print()

print("=" * 80)
print("Top Predictors")
print("=" * 80)

importance = rf_results["tables"]["feature_importance"]

print(
    importance.head(10)
)

###############################################################################
# Display exported files
###############################################################################

print()

print("=" * 80)
print("Exported Robustness Tables")
print("=" * 80)

for name, file in rf_results["exported_files"].items():

    print(f"{name}")

    print(file)

    print()

print("=" * 80)
print("Random Forest robustness analysis completed.")
print("=" * 80)
###############################################################################
# Final summary
###############################################################################

print()

print("=" * 80)
print("REPRODUCTION COMPLETED SUCCESSFULLY")
print("=" * 80)

print()

print("Outputs generated:")

print()

print("Main analysis")

print("  • Hierarchical logistic regression")

print("  • Odds ratios")

print("  • Confidence intervals")

print("  • Correlation matrix")

print("  • VIF")

print("  • Model comparison")

print()

print("Robustness")

print("  • Random Forest")

print("  • Cross-validation")

print("  • ROC-AUC")

print("  • Feature importance")

print("  • Permutation importance")

print("  • Threshold analysis")

print()

print(f"Tables saved to: {TABLES}")

print(f"Robustness results saved to: {ROBUSTNESS}")

print()

print("=" * 80)
