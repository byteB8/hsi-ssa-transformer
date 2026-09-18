"""Choose each model's optimiser on a validation split.

Comparing two architectures under one shared optimiser measures the optimiser as
much as the architecture: HybridSN and the SSA-Transformer have opposite
preferences at 15x15 input. Each model is therefore given its own setting,
chosen the same way for both and scored on data held out of *training*, never on
the test set used to report results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsi_ssat.data import build_split
from hsi_ssat.models import build_model

NEEDS_DEPTH_AXIS = {"hybridsn", "hybridsn_cbam"}
CANDIDATES = [("sgd", 0.01), ("sgd", 0.003), ("adam", 0.001), ("adam", 0.0003)]


def as_tensor(x: np.ndarray, add_depth: bool) -> torch.Tensor:
    t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    return t.unsqueeze(1) if add_depth else t


@torch.no_grad()
def accuracy(model, x: torch.Tensor, y: np.ndarray, device) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(x), 512):
        batch = x[start : start + 512].to(device)
        correct += (model(batch).argmax(1).cpu().numpy() == y[start : start + 512]).sum()
    return float(correct) / len(y)


def evaluate(model_name, split, n_bands, window, optimiser_name, lr, epochs, device):
    add_depth = model_name in NEEDS_DEPTH_AXIS
    loader = DataLoader(
        TensorDataset(as_tensor(split.x_train, add_depth), torch.from_numpy(split.y_train)),
        batch_size=32,
        shuffle=True,
    )
    torch.manual_seed(0)
    model = build_model(model_name, n_classes=split.n_classes, n_bands=n_bands, window=window).to(
        device
    )
    optimiser = (
        torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        if optimiser_name == "sgd"
        else torch.optim.Adam(model.parameters(), lr=lr)
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[epochs // 2], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        scheduler.step()
    return accuracy(model, as_tensor(split.x_val, add_depth), split.y_val, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["hybridsn", "ssa_transformer"])
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--pca-components", type=int, nargs="+", default=[15, 0])
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--val-per-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("reports/optimiser_selection.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selection = {}
    for pca in args.pca_components:
        split = build_split(
            args.data_root,
            window=args.window,
            pca_components=pca,
            protocol="fixed",
            per_class=args.per_class,
            val_per_class=args.val_per_class,
            seed=0,
        )
        n_bands = split.x_train.shape[-1]
        setting = f"w{args.window}_pca{pca}"
        print(f"\n### {setting}: train {len(split.y_train)}, val {len(split.y_val)}")
        for model_name in args.models:
            scores = {}
            for optimiser_name, lr in CANDIDATES:
                value = evaluate(
                    model_name,
                    split,
                    n_bands,
                    args.window,
                    optimiser_name,
                    lr,
                    args.epochs,
                    device,
                )
                scores[f"{optimiser_name}@{lr}"] = value
                print(f"  {model_name:16} {optimiser_name}@{lr:<7} val acc {value * 100:6.2f}")
            best = max(scores, key=scores.get)
            selection[f"{setting}/{model_name}"] = {
                "best": best,
                "val_accuracy": scores[best],
                "all": scores,
            }
            print(f"  -> {model_name}: {best}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selection, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
