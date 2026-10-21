"""Shared, leakage-safe training and evaluation for the churn notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender"]


def load_data(path="Churn_Modeling.csv"):
    """Accept the notebook spelling or Kaggle's original filename."""
    path = Path(path)
    if not path.exists() and path.name == "Churn_Modeling.csv":
        path = path.with_name("Churn_Modelling.csv")
    if not path.exists():
        raise FileNotFoundError("Download the Kaggle CSV first; see README.md.")
    frame = pd.read_csv(path, index_col=0)
    required = set(NUMERIC + CATEGORICAL + ["Exited", "CustomerId"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if frame.CustomerId.isna().any() or frame.CustomerId.duplicated().any():
        raise ValueError("CustomerId must be present and unique before splitting.")
    if frame.Exited.isna().any() or set(frame.Exited.unique()) != {0, 1}:
        raise ValueError("Exited must contain both binary classes and no missing values.")
    return frame[NUMERIC + CATEGORICAL].copy(), frame.Exited.astype(int)


def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


def preprocessing():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="first",
                                 sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("categorical", categorical, CATEGORICAL)])


def model_spec(name):
    if name == "logistic":
        model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=SEED)
        grid = {"model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"]}
    elif name == "forest":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       n_jobs=1, random_state=SEED)
        grid = {"model__max_depth": [5, None], "model__min_samples_leaf": [1, 5]}
    elif name == "boosting":
        model = GradientBoostingClassifier(learning_rate=0.1, random_state=SEED)
        grid = {"model__n_estimators": [100, 200], "model__max_depth": [1, 3]}
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", preprocessing()), ("model", model)]), grid


def tune_model(name, X_train, y_train):
    """Tune only on training rows; preprocessing is refit in each fold."""
    pipeline, grid = model_spec(name)
    search = GridSearchCV(
        pipeline, grid, scoring={"average_precision": "average_precision",
                                 "roc_auc": "roc_auc", "accuracy": "accuracy"},
        refit="average_precision", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        n_jobs=2, error_score="raise", return_train_score=False,
    )
    return search.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    """Evaluate the supplied fitted estimator at the prespecified 0.5 threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
    }
    report = classification_report(y_test, predictions, zero_division=0)
    return metrics, report, probabilities


def evaluate_baseline(X_train, y_train, X_test, y_test):
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    return evaluate_model(baseline, X_test, y_test)


def plot_evaluation(y_test, probabilities):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, probabilities >= 0.5, normalize="true", ax=axes[0])
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[2])
    axes[2].axhline(np.mean(y_test), linestyle="--", color="gray",
                    label="Churn prevalence")
    axes[2].legend()
    fig.tight_layout()
    return fig
