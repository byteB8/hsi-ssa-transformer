# Spectral-Spatial Attention Transformer for Hyperspectral Image Classification

A PyTorch implementation of **SSA-Transformer** (Dang et al., *Computational
Intelligence and Neuroscience*, 2022, [doi:10.1155/2022/7071485][paper]),
evaluated on Pavia University against corrected HybridSN and HybridSN+CBAM
baselines.

This work continues a model survey carried out at **TEXMiN, IIT (ISM) Dhanbad**.

[paper]: https://doi.org/10.1155/2022/7071485

## What this repository corrects

The original notebooks contained a defect that silently prevented the CBAM model
from training at all. Both PyTorch models ended `forward` with

```python
x = torch.softmax(self.fc3(x), dim=1)
return x
```

while the training loop used `nn.CrossEntropyLoss`, which applies `log_softmax`
internally. Softmax was therefore applied **twice**.

The consequence is exact and checkable. If a model emits probabilities `p` and
those are consumed as logits, cross-entropy over `C = 9` classes is bounded:

| model output | resulting loss |
|---|---|
| uniform (`p = 1/9`) | `ln 9` = **2.197** |
| perfectly confident one-hot | `ln(1 + 8/e)` = **1.372** |

No amount of training can push the loss below 1.372. The original CBAM run sat
at **1.89-1.99 for all ten epochs** — squarely inside that band — while the Keras
HybridSN baseline, which correctly paired a softmax output with
`categorical_crossentropy`, reached 96.9% by epoch 2.

Both models here return raw logits. `tests/test_models.py` guards the fix by
overfitting a single batch and asserting the loss falls below the 1.372 floor,
so the bug cannot reappear unnoticed.

A second defect in the original prediction-map cell used `.reshape(1, 1, bands,
h, w)` where a transpose was needed. Reshape reinterprets the buffer rather than
permuting axes, so the published ground-truth-versus-prediction figure did not
correspond to the data. Patch handling now lives in `data.py` and is covered by
tests.

## On the 99.6% figure

The earlier notebook reported 99.63% overall accuracy on Pavia University using a
30/70 random split of all labelled pixels. That number is real but inflated, and
it is worth being precise about why.

Patches are 25x25 and drawn from a single image, so a test patch overlaps its
neighbouring training patches in up to 24 of 25 columns. Train and test share
pixels. This is a known property of the standard protocol rather than a mistake
unique to that notebook — see Nalepa et al., *Validating Hyperspectral Image
Segmentation*, IEEE GRSL 2019 — but it makes cross-paper comparison unreliable.

The default protocol here is the one Dang et al. use: **400 labelled pixels per
class** for training (3,600 of 42,776, about 8%) and everything else for test.
`--protocol ratio` reproduces the older split so the gap between the two can be
measured directly rather than argued about.

## Architecture

Following equations (1)-(8) of the paper:

```
y'    = Conv1(y)            two 3x3 convolutions, spectral depth preserved
y''   = Mse(y') * y'        spectral attention  (SeAM, eq. 2-3)
y'''  = Msa(y'') * y'' + y  spatial attention with residual  (SaAM, eq. 4-5)
y'''' = Conv2(y''')         one 1x1 convolution, C -> k
```

The feature map is cut into non-overlapping `p x p` patches (`p = 3`), each
flattened to a `p*p*k` vector, prepended with a class token and given learned
position codes. Transformer encoder blocks are wired with **dense connections**:
block *i* consumes the projected concatenation of the outputs of all *i*
preceding blocks.

Two quantities are not stated in the paper and are exposed as configuration:
the number of encoder blocks (default 4) and the retained band count `k`
(default 32).

## Layout

```
src/hsi_ssat/
  data.py                  patch extraction, both split protocols
  metrics.py               OA / AA / Kappa
  train.py                 training and evaluation entry point
  models/
    hybridsn.py            HybridSN and its CBAM variant, corrected
    ssa_transformer.py     SSA-Transformer
scripts/run_pavia.sh       the benchmark sweep
tests/                     regression tests
```

## Running

```bash
uv sync
uv run python -m hsi_ssat.train --model ssa_transformer \
    --window 15 --pca-components 0 --protocol fixed --per-class 400 --epochs 80
```

The sweep used for the reported table:

```bash
GROUP=cnn         ./scripts/run_pavia.sh
GROUP=transformer ./scripts/run_pavia.sh
```

Pavia University (`PaviaU.mat`, `PaviaU_gt.mat`) goes in `data/`. It is not
redistributed here.

## Results

Populated by `scripts/run_pavia.sh`; see `reports/`.
