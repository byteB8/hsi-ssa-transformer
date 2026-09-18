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

LABELS = {
    "hybridsn": "HybridSN",
    "hybridsn_cbam": "HybridSN + CBAM",
    "ssa_transformer": "SSA-Transformer",
    "ssa_transformer_nodense": "SSA-Transformer (no dense)",
}


def collect(directories: list[Path]) -> dict[tuple[str, str], list[dict]]:
    grouped = defaultdict(list)
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            key = (record["config"]["model"], record["config"]["protocol"])
            grouped[key].append(record)
    return grouped


def spread(values: list[float]) -> str:
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {statistics.stdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        default=[Path("reports"), Path("reports/seeds")],
    )
    parser.add_argument("--markdown", action="store_true")
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


if __name__ == "__main__":
    main()
