"""Render the figures used in the README.

Every figure is generated from the JSON run records in ``reports/``, so the
plots and the numbers in the text cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "hybridsn": "HybridSN",
    "hybridsn_cbam": "HybridSN + CBAM",
    "ssa_transformer": "SSA-Transformer",
    "ssa_transformer_nodense": "SSA-Transformer\n(no dense)",
}
COLOURS = {
    "hybridsn": "#4C72B0",
    "hybridsn_cbam": "#55A868",
    "ssa_transformer": "#C44E52",
    "ssa_transformer_nodense": "#CCA85B",
}
CLASSES = [
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Painted metal",
    "Bare soil",
    "Bitumen",
    "Bricks",
    "Shadows",
]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )


def load(directories: list[Path]) -> list[dict]:
    records = []
    for directory in directories:
        if directory.exists():
            records += [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    return records


def group_by_model(records, protocol="fixed", window=None, pca=None):
    grouped = defaultdict(list)
    for record in records:
        config = record["config"]
        if config["protocol"] != protocol:
            continue
        if window is not None and config["window"] != window:
            continue
        if pca is not None and config["pca_components"] != pca:
            continue
        grouped[config["model"]].append(record)
    return grouped


def figure_accuracy(records, out: Path) -> None:
    """Overall accuracy per model, mean with one standard deviation."""
    grouped = group_by_model(records)
    order = ["hybridsn", "hybridsn_cbam", "ssa_transformer", "ssa_transformer_nodense"]
    order = [m for m in order if m in grouped]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[1.25, 1])

    means, errors, points = [], [], []
    for model in order:
        values = [r["scores"]["overall_accuracy"] * 100 for r in grouped[model]]
        means.append(statistics.mean(values))
        errors.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        points.append(values)

    positions = np.arange(len(order))
    left.bar(
        positions,
        means,
        yerr=errors,
        capsize=4,
        width=0.6,
        color=[COLOURS[m] for m in order],
        alpha=0.85,
        error_kw={"ecolor": "#333", "lw": 1.2},
    )
    for position, values in zip(positions, points, strict=True):
        left.scatter(
            np.random.default_rng(0).normal(position, 0.045, len(values)),
            values,
            s=12,
            color="#222",
            zorder=3,
            alpha=0.7,
        )
    low = min(min(v) for v in points)
    left.set_ylim(low - 0.25, 100.02)
    left.set_xticks(positions)
    left.set_xticklabels([LABELS[m] for m in order], fontsize=8)
    left.set_ylabel("Overall accuracy (%)")
    left.set_title(
        f"Pavia University, {len(points[0])} seeds\n(dots are individual runs)", fontsize=9
    )

    # Accuracy against parameter count.
    for model in order:
        params = grouped[model][0]["parameters"] / 1e6
        values = [r["scores"]["overall_accuracy"] * 100 for r in grouped[model]]
        right.errorbar(
            params,
            statistics.mean(values),
            yerr=statistics.stdev(values) if len(values) > 1 else 0,
            fmt="o",
            ms=9,
            capsize=4,
            color=COLOURS[model],
            label=LABELS[model].replace("\n", " "),
        )
    right.set_xlabel("Trainable parameters (millions)")
    right.set_ylabel("Overall accuracy (%)")
    right.set_title("Accuracy against model size", fontsize=9)
    right.legend(fontsize=7, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_curves(records, out: Path) -> None:
    """Training loss and accuracy against epoch, averaged over seeds."""
    grouped = group_by_model(records)
    order = [
        m
        for m in ["hybridsn", "hybridsn_cbam", "ssa_transformer", "ssa_transformer_nodense"]
        if m in grouped
    ]

    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 3.8))
    for model in order:
        histories = [r["history"] for r in grouped[model]]
        length = min(len(h) for h in histories)
        epochs = np.arange(1, length + 1)
        losses = np.array([[e["loss"] for e in h[:length]] for h in histories])
        accuracies = np.array([[e["train_accuracy"] * 100 for e in h[:length]] for h in histories])
        for axis, data in ((loss_axis, losses), (accuracy_axis, accuracies)):
            mean = data.mean(axis=0)
            axis.plot(
                epochs, mean, color=COLOURS[model], label=LABELS[model].replace("\n", " "), lw=1.4
            )
            axis.fill_between(
                epochs,
                data.min(axis=0),
                data.max(axis=0),
                color=COLOURS[model],
                alpha=0.15,
                lw=0,
            )
    loss_axis.set_yscale("log")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training loss (log scale)")
    loss_axis.set_title("Convergence", fontsize=9)
    loss_axis.legend(fontsize=7, frameon=False)
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Training accuracy (%)")
    accuracy_axis.set_title("Training accuracy (shaded: min-max over seeds)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_seed_sensitivity(records, out: Path) -> None:
    """How the measured gap between two models moves as seeds accumulate."""
    grouped = group_by_model(records)
    pairs = [
        ("hybridsn", "ssa_transformer", "HybridSN - SSA-Transformer"),
        ("hybridsn", "hybridsn_cbam", "HybridSN - (HybridSN + CBAM)"),
        ("ssa_transformer", "ssa_transformer_nodense", "dense - no dense"),
    ]
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    for left, right, label in pairs:
        if left not in grouped or right not in grouped:
            continue
        a = [
            r["scores"]["overall_accuracy"] * 100
            for r in sorted(grouped[left], key=lambda r: r["config"]["seed"])
        ]
        b = [
            r["scores"]["overall_accuracy"] * 100
            for r in sorted(grouped[right], key=lambda r: r["config"]["seed"])
        ]
        n = min(len(a), len(b))
        counts = range(2, n + 1)
        deltas = [statistics.mean(a[:k]) - statistics.mean(b[:k]) for k in counts]
        axis.plot(list(counts), deltas, marker="o", ms=4, lw=1.4, label=label)
    axis.axhline(0, color="#444", lw=1, ls="--")
    axis.set_xlabel("Seeds averaged")
    axis.set_ylabel("Measured difference in OA (percentage points)")
    axis.set_title("Model gaps shrink toward zero as seeds accumulate", fontsize=9)
    axis.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_per_class(records, out: Path) -> None:
    """Per-class recall for each model."""
    grouped = group_by_model(records)
    order = [
        m
        for m in ["hybridsn", "hybridsn_cbam", "ssa_transformer", "ssa_transformer_nodense"]
        if m in grouped
    ]
    matrix = np.array(
        [
            np.mean([[v * 100 for v in r["scores"]["per_class"]] for r in grouped[m]], axis=0)
            for m in order
        ]
    )
    fig, axis = plt.subplots(figsize=(8.2, 2.9))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=95, vmax=100, aspect="auto")
    axis.set_xticks(range(len(CLASSES)))
    axis.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=7.5)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([LABELS[m].replace("\n", " ") for m in order], fontsize=8)
    axis.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if matrix[i, j] > 98.6 else "#222",
            )
    fig.colorbar(image, ax=axis, label="Recall (%)", pad=0.015)
    axis.set_title("Per-class recall, mean over seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    style()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load([args.reports, args.reports / "seeds"])
    if not records:
        raise SystemExit("no run records found")

    figure_accuracy(records, args.out / "accuracy.png")
    figure_curves(records, args.out / "convergence.png")
    figure_seed_sensitivity(records, args.out / "seed_sensitivity.png")
    figure_per_class(records, args.out / "per_class.png")


if __name__ == "__main__":
    main()
