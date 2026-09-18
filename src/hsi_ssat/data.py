"""Patch extraction and train/test splits for hyperspectral scenes.

Two sampling protocols are supported, and the choice matters more than the model:

``fixed``      Dang et al. (2022) -- a fixed number of labelled pixels per class
               (400 for Pavia University), everything else is test. Training
               touches ~8% of the labelled pixels.
``ratio``      the conventional random split, where a fixed fraction of *all*
               labelled pixels is held out.

Neither protocol removes the overlap between neighbouring patches, so a test
patch always shares pixels with some training patch. ``ratio`` at 70% test makes
that overlap near-total, which is the main reason accuracies above 99% are
routinely reported on this scene. ``fixed`` is the stricter of the two and is the
default here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

SCENES = {
    "pavia_university": ("PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt"),
    "pavia_centre": ("Pavia.mat", "pavia", "Pavia_gt.mat", "pavia_gt"),
    "salinas": ("Salinas_corrected.mat", "salinas_corrected", "Salinas_gt.mat", "salinas_gt"),
}


@dataclass(frozen=True)
class Split:
    """Patch cubes and labels for one fold, plus where each patch came from."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_coords: np.ndarray
    test_coords: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(self.y_train.max()) + 1


def load_scene(root: Path, scene: str) -> tuple[np.ndarray, np.ndarray]:
    if scene not in SCENES:
        raise KeyError(f"unknown scene {scene!r}; expected one of {sorted(SCENES)}")
    image_file, image_key, label_file, label_key = SCENES[scene]
    image = loadmat(root / image_file)[image_key].astype(np.float32)
    labels = loadmat(root / label_file)[label_key].astype(np.int64)
    return image, labels


def standardise(image: np.ndarray) -> np.ndarray:
    """Per-band zero mean, unit variance. Fitted on the whole scene.

    Every published result on these benchmarks normalises this way, so we keep it
    for comparability, but note that it does leak test-set statistics.
    """
    flat = image.reshape(-1, image.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((flat - mean) / std).reshape(image.shape)


def apply_pca(image: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce the spectral axis, then rescale by a single global factor.

    Neither whitening nor per-component standardisation is used. Both give every
    component unit variance, which inflates the low-variance components -- mostly
    sensor noise -- into outliers reaching -145 sigma on this scene. Deep CNNs
    fed inputs like that diverge within a few epochs and settle into an
    all-dead-ReLU state at exactly ln(C).

    Dividing by one scalar keeps the relative importance of the components
    intact while bounding the range (about -13 to +33 here, with 0.1% of values
    beyond 10 sigma).
    """
    if not n_components or n_components >= image.shape[-1]:
        return image
    flat = image.reshape(-1, image.shape[-1])
    reduced = PCA(n_components=n_components, whiten=False).fit_transform(flat)
    reduced = reduced / reduced.std()
    return reduced.reshape(image.shape[0], image.shape[1], n_components).astype(np.float32)


def extract_patches(
    image: np.ndarray, labels: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (patches, labels, coords) for every non-background pixel.

    Labels come back zero-based; background (ground-truth 0) is dropped.
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    margin = window // 2
    padded = np.pad(image, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    rows, cols = np.nonzero(labels)
    patches = np.empty((rows.size, window, window, image.shape[-1]), dtype=np.float32)
    for index, (r, c) in enumerate(zip(rows, cols, strict=True)):
        patches[index] = padded[r : r + window, c : c + window, :]
    targets = labels[rows, cols] - 1
    coords = np.stack([rows, cols], axis=1)
    return patches, targets.astype(np.int64), coords


def _fixed_per_class(targets: np.ndarray, per_class: int, rng: np.random.Generator):
    train_idx = []
    for label in np.unique(targets):
        pool = np.flatnonzero(targets == label)
        take = min(per_class, pool.size)
        if take < per_class:
            # Smaller classes cannot supply the full quota; use all of them and
            # record it rather than silently rebalancing.
            print(f"  class {label}: only {pool.size} samples, using all")
        train_idx.append(rng.choice(pool, size=take, replace=False))
    train_idx = np.concatenate(train_idx)
    mask = np.ones(targets.size, dtype=bool)
    mask[train_idx] = False
    return train_idx, np.flatnonzero(mask)


def _ratio(targets: np.ndarray, test_ratio: float, rng: np.random.Generator):
    train_idx, test_idx = [], []
    for label in np.unique(targets):
        pool = rng.permutation(np.flatnonzero(targets == label))
        cut = int(round(pool.size * (1.0 - test_ratio)))
        train_idx.append(pool[:cut])
        test_idx.append(pool[cut:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def build_split(
    root: Path,
    scene: str = "pavia_university",
    window: int = 15,
    pca_components: int = 0,
    protocol: str = "fixed",
    per_class: int = 400,
    test_ratio: float = 0.7,
    seed: int = 0,
) -> Split:
    image, labels = load_scene(root, scene)
    image = standardise(image)
    image = apply_pca(image, pca_components)
    patches, targets, coords = extract_patches(image, labels, window)

    rng = np.random.default_rng(seed)
    if protocol == "fixed":
        train_idx, test_idx = _fixed_per_class(targets, per_class, rng)
    elif protocol == "ratio":
        train_idx, test_idx = _ratio(targets, test_ratio, rng)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    return Split(
        x_train=patches[train_idx],
        y_train=targets[train_idx],
        x_test=patches[test_idx],
        y_test=targets[test_idx],
        train_coords=coords[train_idx],
        test_coords=coords[test_idx],
    )
