"""Render the full-scene classification map for a trained model.

Patches leave ``extract_patches`` as (N, H, W, bands) and the models expect
(N, bands, H, W). The conversion is a transpose, not a reshape: reshaping
reinterprets the buffer in place and would interleave spectra with spatial
positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hsi_ssat.data import apply_pca, extract_patches, load_scene, standardise
from hsi_ssat.models import build_model

NEEDS_DEPTH_AXIS = {"hybridsn", "hybridsn_cbam"}

DISPLAY_NAMES = {
    "hybridsn": "HybridSN",
    "hybridsn_cbam": "HybridSN + CBAM",
    "ssa_transformer": "SSA-Transformer",
    "ssa_transformer_nodense": "SSA-Transformer (no dense)",
}

# Pavia University palette, background first.
PALETTE = np.array(
    [
        [0, 0, 0],
        [192, 192, 192],
        [0, 255, 0],
        [0, 255, 255],
        [0, 128, 0],
        [255, 0, 255],
        [165, 82, 41],
        [128, 0, 128],
        [255, 0, 0],
        [255, 255, 0],
    ],
    dtype=np.uint8,
)


def colourise(labels: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(labels, 0, len(PALETTE) - 1)]


@torch.no_grad()
def classify_scene(model, image, labels, window, add_depth, device, batch=512):
    patches, _, coords = extract_patches(image, labels, window)
    # (N, H, W, bands) -> (N, bands, H, W): a transpose, never a reshape.
    patches = np.transpose(patches, (0, 3, 1, 2))
    predicted = np.zeros(labels.shape, dtype=np.int64)
    model.eval()
    for start in range(0, len(patches), batch):
        chunk = torch.from_numpy(patches[start : start + batch]).to(device)
        if add_depth:
            chunk = chunk.unsqueeze(1)
        out = model(chunk).argmax(dim=1).cpu().numpy() + 1
        for (r, c), value in zip(coords[start : start + batch], out, strict=True):
            predicted[r, c] = value
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", default="pavia_university")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/map.png"))
    args = parser.parse_args()

    image, labels = load_scene(args.data_root, args.scene)
    image = apply_pca(standardise(image), args.pca_components)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        args.model,
        n_classes=int(labels.max()),
        n_bands=image.shape[-1],
        window=args.window,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    predicted = classify_scene(
        model, image, labels, args.window, args.model in NEEDS_DEPTH_AXIS, device
    )
    agreement = (predicted[labels > 0] == labels[labels > 0]).mean()
    print(f"agreement with ground truth on labelled pixels: {agreement * 100:.2f}%")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is needed for the figure: uv sync --extra viz") from None

    name = DISPLAY_NAMES.get(args.model, args.model)
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    for axis, data, title in (
        (axes[0], labels, "Ground truth"),
        (axes[1], predicted, f"{name}\n{agreement * 100:.2f}% agreement"),
    ):
        axis.imshow(colourise(data))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(
        "Pavia University: full-scene classification",
        fontsize=11,
        y=0.97,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
