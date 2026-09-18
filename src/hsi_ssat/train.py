"""Train and evaluate one model on one hyperspectral scene.

Follows the Dang et al. schedule for Pavia University by default: SGD at 0.01
for 80 epochs, dropping to 0.001 at epoch 41, batch size 32, cross-entropy loss.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hsi_ssat.data import build_split
from hsi_ssat.metrics import score
from hsi_ssat.models import build_model

# HybridSN takes a 5D cube (B, 1, bands, H, W); the transformer takes (B, bands, H, W).
NEEDS_DEPTH_AXIS = {"hybridsn", "hybridsn_cbam"}


@dataclass
class Config:
    model: str = "ssa_transformer"
    scene: str = "pavia_university"
    data_root: str = "data"
    window: int = 15
    pca_components: int = 0
    protocol: str = "fixed"
    per_class: int = 400
    test_ratio: float = 0.7
    epochs: int = 80
    batch_size: int = 32
    optimiser: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_drops: tuple[int, ...] = (41,)
    lr_gamma: float = 0.1
    seed: int = 0
    device: str = "cuda"


def make_loaders(split, config: Config, add_depth: bool):
    def to_tensor(x: np.ndarray) -> torch.Tensor:
        # (N, H, W, bands) -> (N, bands, H, W)
        t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        return t.unsqueeze(1) if add_depth else t

    train = TensorDataset(to_tensor(split.x_train), torch.from_numpy(split.y_train))
    test = TensorDataset(to_tensor(split.x_test), torch.from_numpy(split.y_test))
    return (
        DataLoader(train, batch_size=config.batch_size, shuffle=True, drop_last=False),
        DataLoader(test, batch_size=256, shuffle=False),
    )


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, truth = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        truth.append(labels.numpy())
    return np.concatenate(truth), np.concatenate(preds)


def run(config: Config) -> tuple[dict, nn.Module]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    print(f"building split: {config.scene}, protocol={config.protocol}, window={config.window}")
    split = build_split(
        Path(config.data_root),
        scene=config.scene,
        window=config.window,
        pca_components=config.pca_components,
        protocol=config.protocol,
        per_class=config.per_class,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    n_bands = split.x_train.shape[-1]
    n_classes = split.n_classes
    print(f"  train {len(split.y_train)}  test {len(split.y_test)}  bands {n_bands}")

    add_depth = config.model in NEEDS_DEPTH_AXIS
    train_loader, test_loader = make_loaders(split, config, add_depth)

    model = build_model(
        config.model, n_classes=n_classes, n_bands=n_bands, window=config.window
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model {config.model}: {params:,} trainable parameters on {device}")

    if config.optimiser == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimiser == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"unknown optimiser {config.optimiser!r}")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=list(config.lr_drops), gamma=config.lr_gamma
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        record = {"epoch": epoch, "loss": running / total, "train_accuracy": correct / total}
        history.append(record)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:>3}  loss {record['loss']:.4f}  "
                f"train acc {record['train_accuracy'] * 100:.2f}  "
                f"lr {scheduler.get_last_lr()[0]:g}"
            )

    train_seconds = time.time() - started
    inference_started = time.time()
    y_true, y_pred = predict(model, test_loader, device)
    inference_seconds = time.time() - inference_started
    scores = score(y_true, y_pred, n_classes)
    print(f"\n{config.model}: {scores}  ({train_seconds / 60:.1f} min train)")

    record = {
        "config": asdict(config),
        "parameters": params,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "scores": scores.as_dict(),
        "history": history,
    }
    return record, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name, value in asdict(defaults).items():
        if name == "lr_drops":
            parser.add_argument("--lr-drops", type=int, nargs="*", default=list(value))
        elif isinstance(value, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--out", type=Path, help="write the run record here as JSON")
    parser.add_argument("--checkpoint", type=Path, help="save the trained weights to this path")
    args = parser.parse_args()

    values = {k: v for k, v in vars(args).items() if k not in {"out", "checkpoint"}}
    values["lr_drops"] = tuple(values["lr_drops"])
    config = Config(**values)

    result, model = run(config)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"wrote {args.checkpoint}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
