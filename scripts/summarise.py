"""Aggregate run records into the results table.

Reports OA/AA/Kappa as mean +/- standard deviation across seeds. Single-seed
numbers on this benchmark are not trustworthy on their own: every model scores
above 99%, so gaps of a few tenths sit inside run-to-run variation.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

# Pairs worth testing: each isolates one design decision.
COMPARISONS = [
    ("hybridsn", "hybridsn_cbam", "does CBAM help HybridSN?"),
    ("ssa_transformer", "ssa_transformer_nodense", "does the dense connection help?"),
    ("hybridsn", "ssa_transformer", "transformer vs CNN (different input geometry)"),
]

LABELS = {
    "hybridsn": "HybridSN",
    "hybridsn_cbam": "HybridSN + CBAM",
    "ssa_transformer": "SSA-Transformer",
    "ssa_transformer_nodense": "SSA-Transformer (no dense)",
}


def collect(directories: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """Group run records by (model, protocol).

    Non-run JSON (the optimiser-selection summary) is skipped, and a repeated
    configuration is counted once so a re-run cannot inflate the seed count.
    """
    grouped, seen = defaultdict(list), set()
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            config = record.get("config")
            if config is None:
                continue
            identity = (
                config["model"],
                config["protocol"],
                config["window"],
                config["pca_components"],
                config["seed"],
                config.get("optimiser", "sgd"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            grouped[(config["model"], config["protocol"])].append(record)
    return grouped


def spread(values: list[float]) -> str:
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {statistics.stdev(values):.2f}"


def compare(grouped: dict, alpha: float = 0.05) -> None:
    """Welch t-tests on overall accuracy between paired configurations.

    Every model here scores above 99%, so a difference of a few tenths is only
    meaningful relative to the seed-to-seed spread. Welch is used rather than
    Student because the variances are visibly unequal.
    """
    accuracies = {
        model: [r["scores"]["overall_accuracy"] * 100 for r in records]
        for (model, protocol), records in grouped.items()
        if protocol == "fixed"
    }
    print("\nOverall accuracy, Welch two-sided t-test:")
    for left, right, question in COMPARISONS:
        if left not in accuracies or right not in accuracies:
            continue
        a, b = accuracies[left], accuracies[right]
        if min(len(a), len(b)) < 2:
            print(f"  {question}: not enough seeds")
            continue
        result = stats.ttest_ind(a, b, equal_var=False)
        delta = statistics.mean(a) - statistics.mean(b)
        verdict = (
            "distinguishable"
            if result.pvalue < alpha
            else f"not distinguishable at n={min(len(a), len(b))}"
        )
        print(
            f"  {question}\n"
            f"    {LABELS.get(left, left)} - {LABELS.get(right, right)}: "
            f"delta {delta:+.2f}  p = {result.pvalue:.3f}  -> {verdict}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=[Path("reports"), Path("reports/seeds")],
    )
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--compare", action="store_true", help="run Welch t-tests")
    args = parser.parse_args()

    grouped = collect([d for d in args.dirs if d.exists()])
    if not grouped:
        raise SystemExit("no run records found")

    rows = []
    for (model, protocol), records in sorted(grouped.items()):
        scores = [r["scores"] for r in records]
        rows.append(
            {
                "model": LABELS.get(model, model),
                "protocol": protocol,
                "seeds": len(records),
                "params": records[0]["parameters"],
                "oa": spread([s["overall_accuracy"] * 100 for s in scores]),
                "aa": spread([s["average_accuracy"] * 100 for s in scores]),
                "kappa": spread([s["kappa"] * 100 for s in scores]),
                "minutes": statistics.mean(r["train_seconds"] for r in records) / 60,
            }
        )

    if args.markdown:
        print("| Model | Split | Seeds | Params | OA (%) | AA (%) | Kappa (%) | Train (min) |")
        print("|---|---|---:|---:|---|---|---|---:|")
        for r in rows:
            print(
                f"| {r['model']} | {r['protocol']} | {r['seeds']} | {r['params']:,} | "
                f"{r['oa']} | {r['aa']} | {r['kappa']} | {r['minutes']:.1f} |"
            )
    else:
        header = f"{'model':<28}{'split':<8}{'n':>3}{'params':>11}{'OA':>16}{'AA':>16}{'Kappa':>16}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['model']:<28}{r['protocol']:<8}{r['seeds']:>3}{r['params']:>11,}"
                f"{r['oa']:>16}{r['aa']:>16}{r['kappa']:>16}"
            )

    if args.compare:
        compare(grouped)


if __name__ == "__main__":
    main()
