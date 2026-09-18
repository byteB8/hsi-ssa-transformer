"""Model zoo."""

from __future__ import annotations

from hsi_ssat.models.hybridsn import CBAM, HybridSN
from hsi_ssat.models.ssa_transformer import SSATransformer

__all__ = ["CBAM", "HybridSN", "SSATransformer", "build_model"]


def build_model(name: str, *, n_classes: int, n_bands: int, window: int, **kwargs):
    """Construct a model by name, passing only the arguments it accepts."""
    if name == "hybridsn":
        return HybridSN(n_classes, window=window, n_bands=n_bands, use_cbam=False, **kwargs)
    if name == "hybridsn_cbam":
        return HybridSN(n_classes, window=window, n_bands=n_bands, use_cbam=True, **kwargs)
    if name == "ssa_transformer":
        return SSATransformer(n_classes, n_bands=n_bands, window=window, **kwargs)
    if name == "ssa_transformer_nodense":
        return SSATransformer(n_classes, n_bands=n_bands, window=window, dense=False, **kwargs)
    raise KeyError(f"unknown model {name!r}")
