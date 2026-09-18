"""Split-protocol invariants."""

from __future__ import annotations

import numpy as np
import pytest

from hsi_ssat.data import _fixed_per_class, _ratio, extract_patches


def test_extract_patches_drops_background_and_rebases_labels():
    image = np.random.rand(6, 6, 4).astype(np.float32)
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[1, 1] = 1
    labels[4, 4] = 3
    patches, targets, coords = extract_patches(image, labels, window=3)
    assert patches.shape == (2, 3, 3, 4)
    assert sorted(targets.tolist()) == [0, 2]  # 1 and 3, rebased to zero
    assert coords.tolist() == [[1, 1], [4, 4]]


def test_extract_patches_rejects_even_window():
    image = np.random.rand(4, 4, 2).astype(np.float32)
    labels = np.ones((4, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="odd"):
        extract_patches(image, labels, window=4)


def test_fixed_per_class_is_balanced_and_disjoint():
    targets = np.repeat(np.arange(3), 50)
    train, test = _fixed_per_class(targets, per_class=10, rng=np.random.default_rng(0))
    assert train.size == 30
    assert set(np.bincount(targets[train]).tolist()) == {10}
    assert not set(train) & set(test)
    assert train.size + test.size == targets.size


def test_ratio_split_is_stratified_and_disjoint():
    targets = np.repeat(np.arange(3), 100)
    train, test = _ratio(targets, test_ratio=0.7, rng=np.random.default_rng(0))
    assert not set(train) & set(test)
    assert train.size + test.size == targets.size
    assert set(np.bincount(targets[train]).tolist()) == {30}
