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
    """Read run records, skipping non-run JSON and duplicate configurations."""
    records, seen = [], set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            config = record.get("config")
            if config is None:
                continue  # e.g. the optimiser-selection summary
            key = (
                config["model"], config["protocol"], config["window"],
                config["pca_components"], config["seed"], config.get("optimiser", "sgd"),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
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


def figure_architecture(out: Path) -> None:
    """Schematic of the SSA-Transformer: CNN front end, then dense encoder stack."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 5.2), gridspec_kw={"height_ratios": [1, 0.85]}
    )

    def box(axis, x, y, w, h, label, colour, fontsize=7.4):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4,rounding_size=1.2",
                facecolor=colour,
                edgecolor="#4A5568",
                linewidth=1.1,
            )
        )
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)

    def arrow(axis, x0, y0, x1, y1, colour="#4A5568", style="-|>", conn=None, lw=1.1):
        axis.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle=style,
                mutation_scale=11,
                color=colour,
                linewidth=lw,
                **({"connectionstyle": conn} if conn else {}),
            )
        )

    # --- top: the pipeline ---
    top.set_xlim(0, 118)
    top.set_ylim(0, 26)
    top.axis("off")
    top.grid(False)
    blocks = [
        (1, "Input patch\n15x15x103", "#DDE3EC", 13),
        (16, "Conv1\n2x (3x3)", "#C7D7E8", 10),
        (28, "SeAM\nspectral\nattention", "#9FC0E0", 11),
        (41, "SaAM\nspatial attention\n(+residual)", "#9FC0E0", 13),
        (56, "Conv2\n1x1, C->k", "#C7D7E8", 11),
        (69, "Tokenise\n3x3 patches\n+ CLS + pos", "#E8D6C0", 13),
        (84, "Encoder x4\ndense-connected", "#E0A98C", 15),
        (101, "MLP head\n9 classes", "#DDE3EC", 12),
    ]
    for x, label, colour, width in blocks:
        box(top, x, 8, width, 12, label, colour)
        if x > 1:
            arrow(top, x - 2.1, 14, x - 0.3, 14)
    top.text(
        41,
        3.6,
        "CNN front end: local spectral-spatial features",
        ha="center",
        fontsize=8,
        color="#2C5282",
    )
    top.text(92, 3.6, "Transformer: global context", ha="center", fontsize=8, color="#9C4221")
    top.plot([1, 67], [5.8, 5.8], color="#2C5282", lw=1.2)
    top.plot([69, 113], [5.8, 5.8], color="#9C4221", lw=1.2)

    # --- bottom: how the encoder blocks are wired ---
    bottom.set_xlim(0, 118)
    bottom.set_ylim(0, 40)
    bottom.axis("off")
    bottom.grid(False)
    xs = [14, 40, 66, 92]
    for index, x in enumerate(xs):
        box(bottom, x, 7, 13, 9, f"Encoder {index + 1}\nMHSA + FFN", "#E0A98C", 7.2)
        if index:
            arrow(bottom, x - 1.6, 11.5, x - 0.3, 11.5)
    # Skip links: block i also receives every earlier block's output.
    for source, target, lift in [(0, 2, 0.30), (0, 3, 0.42), (1, 3, 0.24)]:
        arrow(
            bottom,
            xs[source] + 6.5,
            16.2,
            xs[target] + 6.5,
            16.2,
            colour="#B5651D",
            conn=f"arc3,rad=-{lift}",
            lw=1.2,
        )
    bottom.text(
        59,
        24.4,
        "Dense connection: block i consumes a learned projection of all i earlier outputs",
        ha="center",
        fontsize=8,
        color="#B5651D",
        style="italic",
    )
    bottom.text(
        59,
        3.2,
        "concat -> Linear(dim x i -> dim) -> block i",
        ha="center",
        fontsize=7.4,
        color="#666",
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_matched(matched_dir: Path, out: Path) -> None:
    """Architecture comparison once both models receive identical input."""
    from scipy import stats

    records = load([matched_dir])
    if not records:
        print(f"no matched records in {matched_dir}, skipping")
        return

    grouped = defaultdict(list)
    for record in records:
        config = record["config"]
        grouped[(config["pca_components"], config["model"])].append(
            record["scores"]["overall_accuracy"] * 100
        )

    settings = [(15, "15 PCA bands"), (0, "all 103 bands")]
    models = ["hybridsn", "ssa_transformer"]
    fig, axis = plt.subplots(figsize=(7.2, 4))
    width = 0.34
    positions = np.arange(len(settings))
    rng = np.random.default_rng(0)

    for offset, model in zip((-width / 2, width / 2), models, strict=True):
        means = [statistics.mean(grouped[(pca, model)]) for pca, _ in settings]
        errors = [statistics.stdev(grouped[(pca, model)]) for pca, _ in settings]
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            color=COLOURS[model],
            alpha=0.85,
            label=LABELS[model],
            error_kw={"ecolor": "#333", "lw": 1.2},
        )
        for position, (pca, _) in zip(positions, settings, strict=True):
            values = grouped[(pca, model)]
            axis.scatter(
                rng.normal(position + offset, 0.03, len(values)),
                values,
                s=11,
                color="#222",
                zorder=3,
                alpha=0.7,
            )

    for position, (pca, _) in zip(positions, settings, strict=True):
        a, b = grouped[(pca, "hybridsn")], grouped[(pca, "ssa_transformer")]
        pvalue = stats.ttest_ind(a, b, equal_var=False).pvalue
        top = max(max(a), max(b))
        axis.text(
            position,
            top + 0.12,
            f"p = {pvalue:.2f}",
            ha="center",
            fontsize=8,
            color="#444",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in settings])
    axis.set_xlabel("Spectral input (15x15 patches throughout)")
    axis.set_ylabel("Overall accuracy (%)")
    axis.set_ylim(98.9, 100.15)
    axis.set_title(
        "Matched input geometry: neither architecture wins\n"
        "(8 seeds, optimiser chosen per model on validation)",
        fontsize=9,
    )
    axis.legend(fontsize=8, frameon=False, loc="lower right")
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

    figure_architecture(args.out / "architecture.png")
    figure_accuracy(records, args.out / "accuracy.png")
    figure_curves(records, args.out / "convergence.png")
    figure_seed_sensitivity(records, args.out / "seed_sensitivity.png")
    figure_per_class(records, args.out / "per_class.png")
    figure_matched(args.reports / "matched", args.out / "matched_geometry.png")


if __name__ == "__main__":
    main()
