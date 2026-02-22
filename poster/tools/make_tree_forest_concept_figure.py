#!/usr/bin/env python3
"""Create a neuron-dataset concept figure for Decision Tree vs Random Forest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42


def load_neuron_dataset(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    ephys = pd.read_csv(data_dir / "ephys_features.csv")
    cells = pd.DataFrame(json.loads((data_dir / "cells.json").read_text()))

    cells = cells.rename(
        columns={
            "specimen__id": "specimen_id",
            "donor__species": "species",
            "tag__dendrite_type": "dendrite_type",
            "structure_parent__acronym": "brain_area",
        }
    )
    cells["specimen_id"] = cells["specimen_id"].astype(int)

    merged = ephys.merge(
        cells[["specimen_id", "species", "brain_area", "dendrite_type"]],
        on="specimen_id",
        how="inner",
    )
    merged = merged[merged["dendrite_type"].isin(["spiny", "aspiny"])].copy()

    drop_cols = {"id", "specimen_id", "rheobase_sweep_id", "thumbnail_sweep_id", "rheobase_sweep_number"}
    feature_cols = [c for c in ephys.columns if c not in drop_cols]
    return merged, feature_cols


def clean_feature_name(name: str) -> str:
    short = (
        name.replace("_long_square", "_LS")
        .replace("_short_square", "_SS")
        .replace("_ramp", "_R")
        .replace("upstroke_downstroke_ratio", "ud_ratio")
        .replace("input_resistance_mohm", "input_res_mohm")
        .replace("fast_trough", "ftrough")
        .replace("slow_trough", "strough")
    )
    return short


def build_figure(data_dir: Path, out_path: Path) -> None:
    df, feature_cols = load_neuron_dataset(data_dir)
    X = df[feature_cols]
    y = df["dendrite_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Small, interpretable tree for conceptual explanation.
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, random_state=RANDOM_STATE)
    dt.fit(X_train_imp, y_train)
    dt_pred = dt.predict(X_test_imp)
    dt_acc = accuracy_score(y_test, dt_pred)

    # Forest with bagging + random features to illustrate ensemble logic.
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=8,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_imp, y_train)
    rf_pred = rf.predict(X_test_imp)
    rf_acc = accuracy_score(y_test, rf_pred)

    class_names = list(dt.classes_)
    feat_names = [clean_feature_name(n) for n in feature_cols]

    # Pick one example neuron to show forest voting.
    sample_idx = 0
    sample_x = X_test_imp[[sample_idx], :]
    sample_true = y_test.iloc[sample_idx]
    sample_rf_pred = rf.predict(sample_x)[0]
    sample_rf_prob = rf.predict_proba(sample_x)[0]
    tree_votes_raw = np.array([est.predict(sample_x)[0] for est in rf.estimators_])

    # Individual sklearn trees in a forest can emit encoded class ids.
    if np.issubdtype(tree_votes_raw.dtype, np.number):
        rf_classes = list(rf.classes_)
        tree_votes = np.array([rf_classes[int(v)] for v in tree_votes_raw], dtype=object)
    else:
        tree_votes = tree_votes_raw.astype(object)

    vote_counts = pd.Series(tree_votes).value_counts().reindex(class_names, fill_value=0)

    importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)

    fig = plt.figure(figsize=(16.0, 10.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.32, wspace=0.28)

    ax_tree = fig.add_subplot(gs[0, 0])
    plot_tree(
        dt,
        feature_names=feat_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8,
        ax=ax_tree,
    )
    ax_tree.set_title(
        f"Decision Tree Learned from Neuron Dataset\nPredicting dendrite_type (test acc={dt_acc:.3f})",
        fontsize=12,
        fontweight="bold",
    )

    ax_vote = fig.add_subplot(gs[0, 1])
    colors = ["#4c78a8", "#f58518"]
    bars = ax_vote.bar(class_names, vote_counts.values, color=colors, alpha=0.88, edgecolor="#303030")
    for b in bars:
        ax_vote.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1,
            f"{int(b.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_vote.set_ylabel("Number of Trees Voting")
    ax_vote.set_xlabel("Predicted Class")
    ax_vote.set_title("Random Forest Vote Aggregation for One Neuron", fontsize=12, fontweight="bold")
    vote_ymax = max(int(vote_counts.max()), 1)
    ax_vote.set_ylim(0, vote_ymax * 1.22)
    ax_vote.spines["top"].set_visible(False)
    ax_vote.spines["right"].set_visible(False)

    prob_text = " | ".join([f"P({c})={p:.2f}" for c, p in zip(class_names, sample_rf_prob)])
    ax_vote.text(
        0.02,
        0.96,
        (
            f"Example neuron true label: {sample_true}\n"
            f"Forest prediction: {sample_rf_pred}\n"
            f"{prob_text}"
        ),
        transform=ax_vote.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    ax_imp = fig.add_subplot(gs[1, 0])
    importances.sort_values().plot(kind="barh", ax=ax_imp, color="#54a24b")
    ax_imp.set_xlabel("Feature Importance")
    ax_imp.set_title("Top Random Forest Features (Neuron Dataset)", fontsize=12, fontweight="bold")
    ax_imp.spines["top"].set_visible(False)
    ax_imp.spines["right"].set_visible(False)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    dataset_n = len(df)
    n_spiny = int((y == "spiny").sum())
    n_aspiny = int((y == "aspiny").sum())
    text = (
        "Dataset-specific explanation\n\n"
        f"• Dataset: {dataset_n} neurons (spiny={n_spiny}, aspiny={n_aspiny})\n"
        "• Target label: dendrite_type\n"
        "• Input features: electrophysiology summary metrics\n\n"
        "Decision Tree\n"
        "• One model creates a single hierarchy of feature thresholds.\n"
        "• Highly interpretable: each path is an explicit rule.\n\n"
        "Random Forest\n"
        "• 120 trees trained on bootstrap samples and random feature subsets.\n"
        "• Final prediction is an aggregate vote across trees.\n"
        "• Usually improves generalization by reducing variance.\n\n"
        f"Observed on this dataset: tree acc={dt_acc:.3f}, forest acc={rf_acc:.3f}, oob={rf.oob_score_:.3f}"
    )
    ax_text.text(
        0.0,
        1.0,
        text,
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "#f9f9f9", "edgecolor": "#d0d0d0", "alpha": 0.9},
    )

    fig.suptitle(
        "Decision Tree vs Random Forest Using the Neuron Dataset",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a decision-tree vs random-forest concept figure grounded in the neuron dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "scripts/cell_types",
        help="Directory containing ephys_features.csv and cells.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "poster/output/figures/decision_tree_random_forest_concept.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    build_figure(args.data_dir, args.output)
    print(f"Data dir: {args.data_dir}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
