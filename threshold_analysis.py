"""Explore outreach thresholds with nested CV on the training partition only."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from churn_workflow import SEED, load_data, split_data, tune_model


def validate_scores(y, probabilities):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probabilities.shape != y.shape or not len(y):
        raise ValueError("Labels and scores must be nonempty, matching 1D arrays.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Scores must be finite and between zero and one.")
    return y, probabilities


def threshold_table(y, probabilities, thresholds=None):
    """Describe observed classification outcomes; do not assume retention impact."""
    y, probabilities = validate_scores(y, probabilities)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all():
        raise ValueError("Provide a nonempty finite 1D threshold list.")
    if ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Thresholds must be between zero and one.")
    rows = []
    for threshold in thresholds:
        flagged = probabilities >= threshold
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        fn = int(np.sum(~flagged & (y == 1)))
        tn = int(np.sum(~flagged & (y == 0)))
        rows.append({
            "threshold": threshold, "customers": len(y),
            "flagged": tp + fp, "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "contact_rate": (tp + fp) / len(y),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "contacts_per_1000": 1000 * (tp + fp) / len(y),
            "churners_identified_per_1000": 1000 * tp / len(y),
            "nonchurners_contacted_per_1000": 1000 * fp / len(y),
            "churners_missed_per_1000": 1000 * fn / len(y),
        })
    return pd.DataFrame(rows)


def nested_training_scores(name, X_train, y_train, outer_splits=3):
    """Each scored row is excluded from both preprocessing and parameter tuning."""
    if not X_train.index.equals(y_train.index) or not X_train.index.is_unique:
        raise ValueError("Training features and labels need matching unique indices.")
    scores = np.full(len(y_train), np.nan)
    folds = StratifiedKFold(outer_splits, shuffle=True, random_state=SEED)
    for fold, (fit, validation) in enumerate(folds.split(X_train, y_train), start=1):
        print(f"{name}: outer fold {fold}/{outer_splits} (five-fold inner search)", flush=True)
        search = tune_model(name, X_train.iloc[fit], y_train.iloc[fit])
        scores[validation] = search.predict_proba(X_train.iloc[validation])[:, 1]
    validate_scores(y_train, scores)
    return scores


def capacity_table(y, probabilities, fractions=(0.05, 0.1, 0.2)):
    """Rank-only outreach: exact floor(fraction * population), stable tie order.

    Labels describe outcomes after ranking; they never choose which rows to contact.
    """
    y, probabilities = validate_scores(y, probabilities)
    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all():
        raise ValueError("Provide a nonempty finite 1D capacity list.")
    if ((fractions <= 0) | (fractions > 1)).any():
        raise ValueError("Contact fractions must be greater than zero and at most one.")
    ranking = np.argsort(-probabilities, kind="stable")
    prevalence = float(np.mean(y))
    total_churners = int(np.sum(y))
    rows = []
    for fraction in fractions:
        count = int(np.floor(fraction * len(y)))
        tp = int(y[ranking[:count]].sum())
        precision = tp / count if count else 0.0
        cutoff = probabilities[ranking[count - 1]] if count else np.nan
        rows.append({
            "requested_contact_fraction": fraction, "customers": len(y),
            "flagged": count, "contact_rate": count / len(y),
            "true_positives": tp, "false_positives": count - tp,
            "false_negatives": total_churners - tp,
            "precision": precision,
            "recall": tp / total_churners if total_churners else 0.0,
            "lift_over_random": precision / prevalence if prevalence else 0.0,
            "contacts_per_1000": 1000 * count / len(y),
            "churners_identified_per_1000": 1000 * tp / len(y),
            "nonchurners_contacted_per_1000": 1000 * (count - tp) / len(y),
            "minimum_selected_score": cutoff,
            "customers_at_cutoff": int(np.sum(probabilities == cutoff)) if count else 0,
        })
    return pd.DataFrame(rows)


def plot_capacity(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, table in results.groupby("model", sort=False):
        axes[0].plot(table.contact_rate, table.precision, label=name)
        axes[1].plot(table.contact_rate, table.recall, label=name)
    axes[1].plot([0, 0.5], [0, 0.5], "--", color="gray", label="Random ranking")
    for ax, ylabel in zip(axes, ["Precision among contacted customers", "Fraction of churners identified"]):
        ax.set(xlabel="Fraction of customers contacted", ylabel=ylabel, ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Training-only nested CV: rank customers within an outreach budget")
    fig.tight_layout()
    return fig


def plot_tradeoffs(results):
    models = results.model.unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for ax, name in zip(axes[0], models):
        table = results[results.model == name]
        for metric in ["precision", "recall", "contact_rate"]:
            ax.plot(table.threshold, table[metric], label=metric.replace("_", " "))
        ax.set(title=name, xlabel="Score threshold", ylabel="Fraction", ylim=(0, 1))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Training-only nested CV: outreach tradeoffs (not final test results)")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Churn_Modeling.csv")
    parser.add_argument("--output-dir", default="artifacts/thresholds")
    args = parser.parse_args()
    X, y = load_data(args.data)
    X_train, _, y_train, _ = split_data(X, y)
    tables, capacities = [], []
    for name in ["logistic", "forest", "boosting"]:
        scores = nested_training_scores(name, X_train, y_train)
        table = threshold_table(y_train, scores)
        table.insert(0, "model", name)
        tables.append(table)
        capacity = capacity_table(y_train, scores, np.round(np.arange(0.01, 0.51, 0.01), 2))
        capacity.insert(0, "model", name)
        capacities.append(capacity)
    results = pd.concat(tables, ignore_index=True)
    capacity_results = pd.concat(capacities, ignore_index=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "threshold-comparison.csv", index=False)
    capacity_results.to_csv(output / "capacity-comparison.csv", index=False)
    plot_tradeoffs(results).savefig(output / "threshold-tradeoffs.png", dpi=160)
    plot_capacity(capacity_results).savefig(output / "outreach-capacity.png", dpi=160)
    plt.close("all")
    print(capacity_results[capacity_results.requested_contact_fraction.isin([0.05, 0.1, 0.2])][[
        "model", "requested_contact_fraction", "precision", "recall", "lift_over_random", "contacts_per_1000",
        "churners_identified_per_1000", "nonchurners_contacted_per_1000",
    ]].round(3).to_string(index=False))
    print("No policy selected. No final test rows scored. See README for limitations.")


if __name__ == "__main__":
    main()
