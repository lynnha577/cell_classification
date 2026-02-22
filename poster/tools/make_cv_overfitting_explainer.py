#!/usr/bin/env python3
"""Create a methods explainer figure for CV splits and overfitting detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_input_csv(fig_dir: Path, preferred: str | None) -> Path:
    if preferred:
        p = Path(preferred)
        if p.exists():
            return p
        q = fig_dir / preferred
        if q.exists():
            return q
        raise FileNotFoundError(f"Could not find requested CSV: {preferred}")

    candidates = [
        "node_sweep_dendrite_type_electrophysiology.csv",
        "node_sweep_dendrite_type_all_features.csv",
        "node_sweep_dendrite_type_morphology.csv",
    ]
    for name in candidates:
        p = fig_dir / name
        if p.exists():
            return p
    raise FileNotFoundError("No node_sweep_*.csv files found in poster/output/figures")


def find_overfit_start(df_model: pd.DataFrame) -> tuple[int, int]:
    df_model = df_model.sort_values("node").reset_index(drop=True)
    best_idx = int(df_model["test_mean"].idxmax())
    best_node = int(df_model.loc[best_idx, "node"])
    gap = df_model["train_mean"] - df_model["test_mean"]

    # First node past the test-score peak where the train/test gap becomes pronounced.
    for i in range(best_idx + 1, len(df_model)):
        test_drop = float(df_model.loc[best_idx, "test_mean"] - df_model.loc[i, "test_mean"])
        if gap.loc[i] >= 0.03 and test_drop >= 0.005:
            return best_node, int(df_model.loc[i, "node"])

    return best_node, int(df_model["node"].max())


def plot_cv_split_diagram(ax, n_samples: int = 24, n_splits: int = 4) -> None:
    kf = KFold(n_splits=n_splits, shuffle=False)
    mat = np.zeros((n_samples, n_splits), dtype=int)
    for fold, (_, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        mat[test_idx, fold] = 1

    cmap = ListedColormap(["#c7d7f0", "#f4a582"])  # train, test
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("4-Fold Cross-Validation Split Diagram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold Index")
    ax.set_ylabel("Sample Index")
    ax.set_xticks(np.arange(n_splits))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_splits)])
    ax.set_yticks(np.linspace(0, n_samples - 1, 6, dtype=int))
    ax.grid(False)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="None", markersize=10, markerfacecolor="#c7d7f0", label="Train"),
        plt.Line2D([], [], marker="s", linestyle="None", markersize=10, markerfacecolor="#f4a582", label="Test"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)


def plot_model_overfit_panel(
    ax,
    df_model: pd.DataFrame,
    model_label: str,
    n_classes: int,
    y_min: float,
    y_max: float,
) -> None:
    df_model = df_model.sort_values("node")
    x = df_model["node"].to_numpy()
    train = df_model["train_mean"].to_numpy()
    test = df_model["test_mean"].to_numpy()
    train_sem = df_model["train_sem"].to_numpy()
    test_sem = df_model["test_sem"].to_numpy()

    best_node, overfit_start = find_overfit_start(df_model)
    chance = 1.0 / n_classes
    y_range = y_max - y_min

    ax.plot(x, train, color="#1f77b4", lw=2.0, label="Train accuracy")
    ax.plot(x, test, color="#d62728", lw=2.0, linestyle="--", label="Test accuracy")
    ax.fill_between(x, train - train_sem, train + train_sem, color="#1f77b4", alpha=0.12)
    ax.fill_between(x, test - test_sem, test + test_sem, color="#d62728", alpha=0.12)

    ax.axhline(chance, color="gray", linestyle=":", linewidth=1.3, label=f"Chance ({chance:.2f})")
    ax.axvline(best_node, color="#2ca02c", linestyle="-.", linewidth=1.5)
    ax.text(
        best_node + 0.15,
        y_min + 0.03 * y_range,
        f"best test @ {best_node} nodes",
        fontsize=9,
        color="#2ca02c",
    )

    if overfit_start > best_node:
        ax.axvspan(overfit_start, x.max(), color="#ffcccb", alpha=0.24)
        ax.text(
            overfit_start + 0.1,
            y_max - 0.02 * y_range,
            "overfitting region",
            fontsize=9,
            va="top",
            color="#8b0000",
        )

    gap = train - test
    gap_peak_node = int(x[np.argmax(gap)])
    gap_peak = float(np.max(gap))
    ax.text(
        x.min() + 0.2,
        y_min + 0.76 * y_range,
        f"max train-test gap: {gap_peak:.3f} @ {gap_peak_node} nodes",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )

    ax.set_title(model_label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Max Leaf Nodes")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def make_explainer_figure(csv_path: Path, out_path: Path, y_min: float, y_max: float) -> None:
    df = pd.read_csv(csv_path)
    if "n_classes" in df.columns:
        n_classes = int(df["n_classes"].iloc[0])
    else:
        # Dendrite type fallback.
        n_classes = 2

    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], hspace=0.32, wspace=0.18)

    ax_split = fig.add_subplot(gs[0, :])
    plot_cv_split_diagram(ax_split, n_samples=24, n_splits=4)

    ax_dt = fig.add_subplot(gs[1, 0])
    ax_rf = fig.add_subplot(gs[1, 1])
    plot_model_overfit_panel(
        ax_dt,
        df[df["model"] == "DecisionTree"],
        "Decision Tree: Train vs Test Across Nodes",
        n_classes,
        y_min,
        y_max,
    )
    plot_model_overfit_panel(
        ax_rf,
        df[df["model"] == "RandomForest"],
        "Random Forest: Train vs Test Across Nodes",
        n_classes,
        y_min,
        y_max,
    )

    title_stub = csv_path.stem.replace("node_sweep_", "").replace("_", " ")
    fig.suptitle(
        "Cross-Validation Workflow and Overfitting Check\n"
        f"Task/Feature Example: {title_stub}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.02,
        "Interpretation: As node count increases, training accuracy can keep rising while test accuracy plateaus or drops. "
        "A widening train-test gap indicates overfitting.",
        ha="center",
        fontsize=10,
    )

    fig.savefig(out_path, bbox_inches="tight")
    svg_out = out_path.with_suffix(".svg")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CV split + overfitting explainer figure.")
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "poster/output/figures")
    parser.add_argument("--input-csv", type=str, default=None, help="Optional specific node_sweep CSV path or filename.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "poster/output/figures/cv_split_overfitting_explainer.png",
        help="Output PNG path.",
    )
    parser.add_argument("--y-min", type=float, default=0.6, help="Lower y-axis bound for accuracy panels.")
    parser.add_argument("--y-max", type=float, default=1.0, help="Upper y-axis bound for accuracy panels.")
    args = parser.parse_args()

    csv_path = choose_input_csv(args.fig_dir, args.input_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_explainer_figure(csv_path, args.output, args.y_min, args.y_max)
    print(f"Input CSV: {csv_path}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
