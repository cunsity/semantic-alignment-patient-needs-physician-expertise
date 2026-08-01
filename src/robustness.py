"""
robustness.py

Random Forest robustness analyses for

Patient–Physician Fit in Online Healthcare Platforms:
Measuring Symptom–Specialty Alignment with Large
Language Model Embeddings.

Author:
Isaac Owusu Asante
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


###############################################################################
# Helper functions
###############################################################################

def ensure_directory(directory):

    """
    Create output directory if necessary.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


###############################################################################
# Data preparation
###############################################################################

def prepare_random_forest_data(
    df,
    predictors,
    outcome="lower_rating",
    categorical_variables=None,
    test_size=0.20,
    random_state=42,
):
    """
    Prepare Random Forest data.

    Automatically one-hot encodes categorical predictors.
    """

    if categorical_variables is None:
        categorical_variables = []

    x = df[predictors].copy()

    y = df[outcome].copy()

    if categorical_variables:

        x = pd.get_dummies(
            x,
            columns=categorical_variables,
            drop_first=True,
            dtype=float,
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


###############################################################################
# Random Forest estimation
###############################################################################

def fit_random_forest(
    x_train,
    y_train,
    random_state=42,
):
    """
    Fit Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


###############################################################################
# Model evaluation
###############################################################################

def evaluate_random_forest(
    model,
    x_test,
    y_test,
):
    """
    Evaluate prediction performance.
    """

    predictions = model.predict(
        x_test,
    )

    probabilities = model.predict_proba(
        x_test,
    )[:, 1]

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions,
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
            ),

        "F1":
            f1_score(
                y_test,
                predictions,
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities,
            ),
    }

    confusion = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    return (
        metrics,
        confusion,
        pd.DataFrame(report).T,
    )
###############################################################################
# Feature importance
###############################################################################

def feature_importance_table(
    model,
    feature_names,
    decimals=4,
):
    """
    Extract impurity-based Random Forest feature importance.

    Parameters
    ----------
    model
        Fitted RandomForestClassifier.

    feature_names
        Predictor names in the same order used to fit the model.

    decimals
        Number of decimal places.

    Returns
    -------
    pandas.DataFrame
        Feature importance table sorted from highest to lowest.
    """

    importance = pd.DataFrame(
        {
            "Feature": list(feature_names),
            "Importance": model.feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance["Rank"] = (
        np.arange(
            1,
            len(importance) + 1,
        )
    )

    importance["Importance"] = (
        importance["Importance"]
        .round(decimals)
    )

    return importance[
        [
            "Rank",
            "Feature",
            "Importance",
        ]
    ]


###############################################################################
# Permutation importance
###############################################################################

def permutation_importance_table(
    model,
    x_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    decimals=4,
):
    """
    Compute permutation importance on the held-out test sample.

    Permutation importance evaluates the decrease in predictive
    performance after randomly shuffling each predictor.

    Parameters
    ----------
    model
        Fitted classifier.

    x_test
        Test-set predictors.

    y_test
        Test-set outcome.

    scoring
        Evaluation metric used to assess performance reduction.

    n_repeats
        Number of shuffling repetitions per predictor.

    random_state
        Random seed.

    decimals
        Number of decimal places.

    Returns
    -------
    pandas.DataFrame
        Mean and standard deviation of permutation importance.
    """

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator=model,
        X=x_test,
        y=y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "Feature": x_test.columns,
            "Mean Importance": result.importances_mean,
            "SD Importance": result.importances_std,
        }
    )

    importance = (
        importance
        .sort_values(
            by="Mean Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance["Rank"] = (
        np.arange(
            1,
            len(importance) + 1,
        )
    )

    importance["Mean Importance"] = (
        importance["Mean Importance"]
        .round(decimals)
    )

    importance["SD Importance"] = (
        importance["SD Importance"]
        .round(decimals)
    )

    return importance[
        [
            "Rank",
            "Feature",
            "Mean Importance",
            "SD Importance",
        ]
    ]


###############################################################################
# Cross-validation
###############################################################################

def cross_validate_random_forest(
    model,
    x,
    y,
    n_splits=5,
    random_state=42,
):
    """
    Evaluate the Random Forest using stratified cross-validation.

    The function reports:

    - accuracy
    - precision
    - recall
    - F1-score
    - ROC-AUC

    Parameters
    ----------
    model
        RandomForestClassifier or compatible estimator.

    x
        Predictor matrix.

    y
        Binary outcome.

    n_splits
        Number of stratified folds.

    random_state
        Random seed used for fold allocation.

    Returns
    -------
    tuple
        Raw fold-level scores and a summary table.
    """

    from sklearn.model_selection import (
        StratifiedKFold,
        cross_validate,
    )

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        estimator=model,
        X=x,
        y=y,
        cv=cross_validator,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    metric_map = {
        "test_accuracy": "Accuracy",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1",
        "test_roc_auc": "ROC-AUC",
    }

    summary_rows = []

    for score_name, display_name in metric_map.items():

        values = scores[score_name]

        summary_rows.append(
            {
                "Metric": display_name,
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "Minimum": values.min(),
                "Maximum": values.max(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    numeric_columns = [
        "Mean",
        "SD",
        "Minimum",
        "Maximum",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(4)
    )

    fold_scores = pd.DataFrame(
        {
            "Fold": np.arange(
                1,
                n_splits + 1,
            ),
            "Accuracy": scores["test_accuracy"],
            "Precision": scores["test_precision"],
            "Recall": scores["test_recall"],
            "F1": scores["test_f1"],
            "ROC-AUC": scores["test_roc_auc"],
        }
    )

    fold_scores[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    ] = fold_scores[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    ].round(4)

    return fold_scores, summary
###############################################################################
# ROC curve data
###############################################################################

def roc_curve_table(
    model,
    x_test,
    y_test,
    decimals=6,
):
    """
    Generate ROC-curve coordinates for a fitted classifier.

    Parameters
    ----------
    model
        Fitted classifier supporting predict_proba().

    x_test
        Test-set predictors.

    y_test
        Test-set binary outcome.

    decimals
        Number of decimal places used in the returned table.

    Returns
    -------
    pandas.DataFrame
        False-positive rates, true-positive rates, and thresholds.

    float
        ROC-AUC score.
    """

    from sklearn.metrics import (
        roc_curve,
        roc_auc_score,
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y_test,
            probabilities,
        )
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities,
    )

    roc_table = pd.DataFrame(
        {
            "False Positive Rate": false_positive_rate,
            "True Positive Rate": true_positive_rate,
            "Threshold": thresholds,
        }
    )

    roc_table = roc_table.round(decimals)

    return roc_table, auc_score


###############################################################################
# Confusion matrix table
###############################################################################

def confusion_matrix_table(
    model,
    x_test,
    y_test,
):
    """
    Produce a labelled confusion matrix.

    Rows represent actual classes.
    Columns represent predicted classes.
    """

    predictions = model.predict(
        x_test
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    table = pd.DataFrame(
        matrix,
        index=[
            "Actual 0",
            "Actual 1",
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
        ],
    )

    return table


###############################################################################
# Prediction output
###############################################################################

def prediction_table(
    model,
    x_test,
    y_test,
    include_features=False,
):
    """
    Create observation-level prediction results.

    Parameters
    ----------
    model
        Fitted classifier.

    x_test
        Test-set predictors.

    y_test
        Test-set binary outcome.

    include_features
        Whether to include predictor values in the exported table.

    Returns
    -------
    pandas.DataFrame
        Actual class, predicted class, and predicted probability.
    """

    predictions = model.predict(
        x_test
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    output = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions,
            "Probability_lower_rating": probabilities,
        },
        index=x_test.index,
    )

    if include_features:

        output = pd.concat(
            [
                output,
                x_test,
            ],
            axis=1,
        )

    return output.sort_index()


###############################################################################
# Threshold-based evaluation
###############################################################################

def threshold_evaluation_table(
    model,
    x_test,
    y_test,
    thresholds=None,
    decimals=4,
):
    """
    Evaluate classification performance across probability thresholds.

    This is useful when the default threshold of 0.50 may not provide
    the desired balance between precision and recall.

    Parameters
    ----------
    model
        Fitted classifier.

    x_test
        Test-set predictors.

    y_test
        Test-set outcome.

    thresholds
        Iterable of probability thresholds. Defaults to values from
        0.10 to 0.90.

    decimals
        Number of decimal places.

    Returns
    -------
    pandas.DataFrame
        Accuracy, precision, recall, specificity, and F1 by threshold.
    """

    if thresholds is None:

        thresholds = np.arange(
            0.10,
            0.91,
            0.05,
        )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        rows.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "Specificity": specificity,
                "F1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    table = pd.DataFrame(rows)

    numeric_columns = [
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
    ]

    table[numeric_columns] = (
        table[numeric_columns]
        .round(decimals)
    )

    return table


###############################################################################
# Summary metrics table
###############################################################################

def random_forest_metrics_table(
    metrics,
    decimals=4,
):
    """
    Convert the evaluation-metrics dictionary into a DataFrame.
    """

    rows = [
        {
            "Metric": metric,
            "Value": value,
        }
        for metric, value in metrics.items()
    ]

    table = pd.DataFrame(rows)

    table["Value"] = (
        table["Value"]
        .round(decimals)
    )

    return table
###############################################################################
# Export utilities
###############################################################################

def export_robustness_tables(
    tables,
    output_directory="results/robustness",
):
    """
    Export robustness-analysis tables to Excel files.

    Parameters
    ----------
    tables
        Dictionary whose values are pandas DataFrames.

    output_directory
        Folder in which the exported files will be saved.

    Returns
    -------
    dict
        Mapping from table names to exported file paths.
    """

    output_directory = ensure_directory(
        output_directory
    )

    exported_files = {}

    for table_name, table in tables.items():

        file_path = (
            output_directory
            / f"{table_name}.xlsx"
        )

        table.to_excel(
            file_path,
            index=True,
        )

        exported_files[table_name] = file_path

    return exported_files


###############################################################################
# Full Random Forest robustness workflow
###############################################################################

def run_random_forest_robustness(
    df,
    predictors,
    outcome="lower_rating",
    categorical_variables=None,
    test_size=0.20,
    random_state=42,
    n_estimators=500,
    cross_validation_folds=5,
    permutation_repeats=10,
    output_directory="results/robustness",
    export=True,
):
    """
    Run the complete Random Forest robustness analysis.

    The workflow:

    1. selects complete cases;
    2. creates a stratified train/test split;
    3. fits a Random Forest classifier;
    4. evaluates held-out prediction performance;
    5. computes impurity-based feature importance;
    6. computes permutation importance;
    7. performs stratified cross-validation;
    8. generates ROC-curve coordinates;
    9. creates confusion-matrix and prediction tables;
    10. evaluates alternative classification thresholds;
    11. optionally exports all tables to Excel.

    Parameters
    ----------
    df
        Analysis DataFrame.

    predictors
        Predictor column names.

    outcome
        Binary outcome column.

    test_size
        Proportion allocated to the held-out test sample.

    random_state
        Random seed.

    n_estimators
        Number of trees in the Random Forest.

    cross_validation_folds
        Number of stratified cross-validation folds.

    permutation_repeats
        Number of repetitions for permutation importance.

    output_directory
        Folder for exported robustness tables.

    export
        Whether to export the generated tables.

    Returns
    -------
    dict
        Fitted model, train/test data, evaluation metrics,
        diagnostic tables, and exported file paths.
    """

    required_columns = [
        outcome,
        *predictors,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing variables required for Random Forest analysis: "
            f"{missing_columns}"
        )

    analysis_df = (
        df[required_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    if analysis_df.empty:

        raise ValueError(
            "No observations remain after complete-case filtering."
        )

    outcome_values = set(
        analysis_df[outcome].unique()
    )

    if not outcome_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"Outcome variable '{outcome}' must contain only 0 and 1."
        )

    if len(outcome_values) != 2:

        raise ValueError(
            f"Outcome variable '{outcome}' must contain both classes."
        )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = prepare_random_forest_data(
        df=analysis_df,
        predictors=predictors,
        outcome=outcome,
        categorical_variables=categorical_variables,
        test_size=test_size,
        random_state=random_state,
    )
    x = pd.concat(
        [
            x_train,
            x_test,
        ]
    ).sort_index()
    y = pd.concat(
        [
            y_train,
            y_test,
        ]
    ).sort_index()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    (
        metrics,
        confusion,
        classification,
    ) = evaluate_random_forest(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    metrics_table = random_forest_metrics_table(
        metrics
    )

    impurity_importance = feature_importance_table(
        model=model,
        feature_names=x_train.columns,
    )

    permutation_importance = permutation_importance_table(
        model=model,
        x_test=x_test,
        y_test=y_test,
        scoring="roc_auc",
        n_repeats=permutation_repeats,
        random_state=random_state,
    )

    (
        cross_validation_scores,
        cross_validation_summary,
    ) = cross_validate_random_forest(
        model=model,
        x=x,
        y=y,
        n_splits=cross_validation_folds,
        random_state=random_state,
    )

    roc_data, roc_auc = roc_curve_table(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    labelled_confusion = confusion_matrix_table(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    predictions = prediction_table(
        model=model,
        x_test=x_test,
        y_test=y_test,
        include_features=False,
    )

    threshold_results = threshold_evaluation_table(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    split_summary = pd.DataFrame(
        [
            {
                "Sample": "Full analysis sample",
                "Observations": len(analysis_df),
                "Positive cases": int(y.sum()),
                "Negative cases": int(
                    len(y) - y.sum()
                ),
            },
            {
                "Sample": "Training sample",
                "Observations": len(y_train),
                "Positive cases": int(
                    y_train.sum()
                ),
                "Negative cases": int(
                    len(y_train) - y_train.sum()
                ),
            },
            {
                "Sample": "Test sample",
                "Observations": len(y_test),
                "Positive cases": int(
                    y_test.sum()
                ),
                "Negative cases": int(
                    len(y_test) - y_test.sum()
                ),
            },
        ]
    )

    roc_summary = pd.DataFrame(
        [
            {
                "Metric": "ROC-AUC",
                "Value": roc_auc,
            }
        ]
    )

    roc_summary["Value"] = (
        roc_summary["Value"]
        .round(4)
    )

    raw_confusion = pd.DataFrame(
        confusion,
        index=[
            "Actual 0",
            "Actual 1",
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
        ],
    )

    tables = {
        "sample_split_summary":
            split_summary,

        "random_forest_metrics":
            metrics_table,

        "classification_report":
            classification,

        "confusion_matrix":
            labelled_confusion,

        "raw_confusion_matrix":
            raw_confusion,

        "feature_importance":
            impurity_importance,

        "permutation_importance":
            permutation_importance,

        "cross_validation_scores":
            cross_validation_scores,

        "cross_validation_summary":
            cross_validation_summary,

        "roc_curve":
            roc_data,

        "roc_summary":
            roc_summary,

        "threshold_evaluation":
            threshold_results,

        "predictions":
            predictions,
    }

    exported_files = {}

    if export:

        exported_files = export_robustness_tables(
            tables=tables,
            output_directory=output_directory,
        )

    return {
        "model": model,

        "analysis_data": analysis_df,

        "x_train": x_train,

        "x_test": x_test,

        "y_train": y_train,

        "y_test": y_test,

        "metrics": metrics,

        "roc_auc": roc_auc,

        "tables": tables,

        "exported_files": exported_files,
    }


###############################################################################
# End of module
###############################################################################
