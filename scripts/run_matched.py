"""Compare architectures at matched input geometry.

The main benchmark gives each model the input it was designed for, so any
difference between them confounds architecture with spatial context and spectral
preprocessing. Here both models receive identical input, at two settings so the
conclusion does not rest on one choice of geometry:

    w15_pca15   15x15 patches, 15 PCA bands
    w15_pca0    15x15 patches, all 103 bands

15 is the largest window below HybridSN's native 25 that stays divisible by the
transformer's patch size of 3.

Each model uses the optimiser chosen for it on the validation split by
``select_optimiser.py``; forcing one shared optimiser would measure the optimiser
rather than the architecture, and the two models do not agree on one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("reports/optimiser_selection.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/matched"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/matched"))
    parser.add_argument("--models", nargs="+", default=["hybridsn", "ssa_transformer"])
    parser.add_argument("--settings", nargs="+", default=["w15_pca15", "w15_pca0"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    for setting in args.settings:
        pca = int(setting.split("pca")[1])
        for model in args.models:
            key = f"{setting}/{model}"
            if key not in selection:
                raise SystemExit(f"no optimiser recorded for {key}")
            optimiser, lr = selection[key]["best"].split("@")
            for seed in args.seeds:
                tag = f"{model}_{setting}_s{seed}"
                record = args.out_dir / f"{tag}.json"
                if record.exists():
                    print(f"skip {tag}")
                    continue
                command = [
                    args.python,
                    "-m",
                    "hsi_ssat.train",
                    "--model",
                    model,
                    "--window",
                    "15",
                    "--pca-components",
                    str(pca),
                    "--protocol",
                    "fixed",
                    "--per-class",
                    "400",
                    "--optimiser",
                    optimiser,
                    "--lr",
                    lr,
                    "--epochs",
                    str(args.epochs),
                    "--seed",
                    str(seed),
                    "--out",
                    str(record),
                ]
                log = args.log_dir / f"{tag}.log"
                with log.open("w") as handle:
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
                tail = [line for line in log.read_text().splitlines() if line.startswith(model)]
                print(f"{tag:42} {optimiser}@{lr:<7} {tail[-1] if tail else 'no result'}")


if __name__ == "__main__":
    main()
