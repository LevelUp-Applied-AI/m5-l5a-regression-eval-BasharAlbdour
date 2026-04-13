"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import os
import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """

    import os

    possible_paths = [
        filepath,
        os.path.join("starter", filepath),
        os.path.join("starter", "data", "telecom_churn.csv"),
        os.path.join("data", "telecom_churn.csv"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break

    df = pd.read_csv(filepath)
    print(f"shape: {df.shape}")
    print(f"missing: {df.isnull().sum()}")
    print(f"churn distribution:\n{df['churned'].value_counts()}")
    print(f"Churn rate: {df['churned'].mean():.2%}")

    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    stratify = None if y.dtype == float else y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(f"Split (target='{target_col}')")
    print(f"X_train: {X_train.shape} , X_test: {X_test.shape}")
    if stratify is not None:
        print(f"Churn rate - train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=42, max_iter=1000, class_weight="balanced"
                ),
            ),
        ]
    )
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.title("Logistic Regression - Confusion Matrix")
    # plt.tight_layout()
    # plt.savefig("confusion_matrix.png")
    # plt.close()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return metrics


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0))])
    return pipeline


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Ridge Regression ===")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")

    return {"mae": mae, "r2": r2}


def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression.

    Returns:
        sklearn Pipeline object.
    """
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", Lasso(alpha=0.1))])
    return pipeline


def compare_ridge_lasso(ridge_pipe, lasso_pipe, X_train, y_train, feature_names):
    """Fit both pipelines and print coefficients side by side."""
    ridge_pipe.fit(X_train, y_train)
    lasso_pipe.fit(X_train, y_train)

    ridge_coefs = ridge_pipe.named_steps["regressor"].coef_
    lasso_coefs = lasso_pipe.named_steps["regressor"].coef_

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Ridge": ridge_coefs,
            "Lasso": lasso_coefs,
        }
    )

    print("\n=== Ridge vs Lasso Coefficients ===")
    print(coef_df.to_string(index=False))

    zeroed = coef_df[coef_df["Lasso"] == 0]["Feature"].tolist()
    print(f"\nFeatures driven to zero by Lasso: {zeroed}")
    # Features zeroed by Lasso have weak linear signal for predicting monthly_charges.
    # Lasso's L1 penalty shrinks uninformative coefficients all the way to zero,
    # effectively performing automatic feature selection.


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy"
    )
    print("\n=== Score of each fold ===")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    print(f"  Mean: {scores.mean():.4f}  |  Std: {scores.std():.4f}")
    return scores


# ==============================================================================
# Task 7 - Summary of Findings (not autograder-tested)
# ==============================================================================
# SUMMARY OF FINDINGS
#
# Dataset: 1500 customers, 13 columns, 16.27% churn rate (244 churners out of 1500).
#
# 1. Most important features for predicting churn
#    - tenure:            Strongest predictor. Ridge coefficient of -26.17 shows
#                         longer-tenured customers are far less likely to churn.
#    - total_charges:     Large positive coefficient (35.30) — customers with higher
#                         cumulative bills are more likely to churn.
#    - num_support_calls: Positive coefficient (0.51) — more support calls signal
#                         frustration and predict churn.
#    - senior_citizen:    Positive coefficient (0.73) — seniors churn at a higher rate.
#    - has_partner / has_dependents: Smaller positive coefficients in this dataset,
#                         suggesting less "stickiness" effect than expected.
#
# 2. Model performance and precision vs recall trade-off
#    - The model achieved 63% accuracy, 23% precision, and 51% recall on the test set.
#    - class_weight="balanced" deliberately sacrifices precision to boost recall:
#      the model catches ~51% of actual churners but raises many false alarms
#      (only 1 in 4 flagged customers actually churns).
#    - RECALL is the more important metric here. Missing a churner (false negative)
#      means losing a customer with no chance to intervene. A false positive
#      (wrongly flagging a loyal customer) only costs a small retention offer.
#    - Accuracy is misleading on this imbalanced dataset: a naive model predicting
#      "no churn" always would score ~84% accuracy while catching zero churners.
#      F1 (0.31) and recall are the meaningful metrics to track.
#    - Cross-validation confirmed stable generalization: mean accuracy 0.607 +/- 0.019
#      across 5 folds, showing the model is not overfitting.
#
# 3. Ridge vs Lasso comparison (monthly_charges regression, R2=0.71, MAE=10.61)
#    - Ridge and Lasso produced very similar coefficients on this dataset.
#    - Lasso (alpha=0.1) did NOT drive any features to zero, meaning all six features
#      carry linear signal for predicting monthly_charges.
#    - tenure and total_charges dominate, which makes sense since monthly charges
#      accumulate into total charges over the tenure period.
#    - R2=0.71 means the model explains 71% of variance in monthly charges.
#
# 4. Recommended next steps
#    - Try ensemble models (Random Forest, Gradient Boosting / XGBoost) which often
#      outperform logistic regression on tabular data.
#    - Use ROC-AUC or F1 as the CV scoring metric instead of accuracy.
#    - Incorporate categorical features (contract_type, internet_service,
#      payment_method) which were excluded here but likely carry strong signal.
#    - Tune the classification threshold below 0.5 using a precision-recall curve
#      to find the operating point that best balances business costs.
#    - Compare against a DummyClassifier baseline (Thursday's Integration Task) to
#      confirm the model adds real value over a naive majority-class strategy.

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "num_support_calls",
            "senior_citizen",
            "has_partner",
            "has_dependents",
        ]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[
            [
                "tenure",
                "total_charges",
                "num_support_calls",
                "senior_citizen",
                "has_partner",
                "has_dependents",
                "monthly_charges",
            ]
        ].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")

        # Task 5 - Lasso comparison (not autograder-tested)
        reg_features = [
            "tenure",
            "total_charges",
            "num_support_calls",
            "senior_citizen",
            "has_partner",
            "has_dependents",
        ]
        compare_ridge_lasso(
            build_ridge_pipeline(), build_lasso_pipeline(), X_tr, y_tr, reg_features
        )
