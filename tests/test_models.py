"""Guards against the defects found in the original TEXMiN notebooks."""

from __future__ import annotations

import math

import pytest
import torch

from hsi_ssat.models import build_model

MODELS = {
    "hybridsn": ((2, 1, 15, 25, 25), {"n_bands": 15, "window": 25}),
    "hybridsn_cbam": ((2, 1, 15, 25, 25), {"n_bands": 15, "window": 25}),
    "ssa_transformer": ((2, 103, 15, 15), {"n_bands": 103, "window": 15}),
    "ssa_transformer_nodense": ((2, 103, 15, 15), {"n_bands": 103, "window": 15}),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_output_shape(name):
    shape, kwargs = MODELS[name]
    model = build_model(name, n_classes=9, **kwargs)
    assert model(torch.randn(*shape)).shape == (shape[0], 9)


@pytest.mark.parametrize("name", list(MODELS))
def test_returns_logits_not_probabilities(name):
    """The original notebooks ended forward() with softmax and then fed the
    result to CrossEntropyLoss, applying softmax twice. Rows summing to one are
    the signature of that bug."""
    shape, kwargs = MODELS[name]
    model = build_model(name, n_classes=9, **kwargs)
    out = model(torch.randn(*shape))
    row_sums = out.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert (out < 0).any(), "logits should be able to go negative"


@pytest.mark.parametrize("name", list(MODELS))
def test_loss_can_fall_below_double_softmax_floor(name):
    """With softmax applied twice, cross-entropy on nine classes cannot drop
    below ln(1 + 8/e) = 1.372 no matter how confident the model is. Overfitting
    a single batch must beat that floor."""
    floor = math.log(1 + 8 / math.e)
    shape, kwargs = MODELS[name]
    torch.manual_seed(0)
    model = build_model(name, n_classes=9, **kwargs)
    images = torch.randn(*shape)
    labels = torch.tensor([0, 1])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loss = torch.tensor(float("inf"))
    for _ in range(60):
        loss = criterion(model(images), labels)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert loss.item() < floor, f"{name} stuck at {loss.item():.3f} (floor {floor:.3f})"


def test_dense_connection_adds_parameters():
    dense = build_model("ssa_transformer", n_classes=9, n_bands=103, window=15)
    plain = build_model("ssa_transformer_nodense", n_classes=9, n_bands=103, window=15)
    n_dense = sum(p.numel() for p in dense.parameters())
    n_plain = sum(p.numel() for p in plain.parameters())
    assert n_dense > n_plain
