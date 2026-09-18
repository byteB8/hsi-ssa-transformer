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

Pavia University. `fixed` is the Dang et al. protocol (400 labelled pixels per
class, 3,600 train / 39,176 test); `ratio` is the older 30/70 random split.
Eight seeds per configuration, mean +/- standard deviation.

| Model | Split | Seeds | Params | OA (%) | AA (%) | Kappa (%) | Train (min) |
|---|---|---:|---:|---|---|---|---:|
| HybridSN | fixed | 8 | 4,844,793 | 99.56 ± 0.32 | 99.62 ± 0.21 | 99.40 ± 0.44 | 1.0 |
| HybridSN | ratio | 1 | 4,844,793 | 99.97 | 99.87 | 99.96 | 3.5 |
| HybridSN + CBAM | fixed | 8 | 4,845,489 | 99.59 ± 0.14 | 99.68 ± 0.08 | 99.44 ± 0.19 | 2.5 |
| SSA-Transformer | fixed | 8 | 3,374,019 | 99.40 ± 0.14 | 99.49 ± 0.06 | 99.17 ± 0.19 | 4.1 |
| SSA-Transformer (no dense) | fixed | 8 | 2,875,491 | 99.34 ± 0.15 | 99.45 ± 0.12 | 99.09 ± 0.21 | 4.0 |

Reproduce with `python scripts/summarise.py --compare`.

### What is and is not distinguishable

Every configuration scores above 99%, so the interesting question is not which
number is largest but which gaps survive the seed-to-seed spread. Welch
two-sided t-tests on overall accuracy, n = 8:

| Comparison | Delta OA | p | Verdict |
|---|---:|---:|---|
| HybridSN vs HybridSN + CBAM | -0.03 | 0.830 | not distinguishable |
| SSA-Transformer, dense vs not | +0.06 | 0.419 | not distinguishable |
| HybridSN vs SSA-Transformer | +0.17 | 0.216 | not distinguishable |

**None of the three survive.** That includes the paper's headline claim for the
dense connection, which this reimplementation cannot confirm on Pavia University
at this sample size.

This is worth stating carefully, because at three seeds the picture looked
different and more flattering:

| Comparison | n = 3 | n = 8 |
|---|---|---|
| HybridSN vs SSA-Transformer | +0.36, p = 0.006 **"significant"** | +0.17, p = 0.216 |
| HybridSN vs HybridSN + CBAM | +0.24, p = 0.126 | -0.03, p = 0.830 |
| dense vs not | +0.12, p = 0.500 | +0.06, p = 0.419 |

The one result that reached significance at n = 3 evaporated at n = 8, and the
CBAM gap changed sign. Three seeds were not enough to say anything here, and
single-seed numbers -- the norm in the original notebooks -- are not evidence at
all. Dang et al. average ten repeats for this reason.

No run diverged: final training loss lies between 0.000 and 0.005 everywhere, so
the spread is genuine variation in generalisation rather than unstable training.

### An open question

HybridSN's spread is more than twice CBAM's (sd 0.32 vs 0.14, a 5.4x variance
ratio), which would suggest CBAM buys stability rather than accuracy. The
evidence is not conclusive: Bartlett's test gives p = 0.040 but Levene's gives
p = 0.356, and Levene is the more trustworthy of the two here because it does
not assume normality and the HybridSN samples contain a low outlier. Resolving
this needs more seeds.

### On the protocol gap

HybridSN scores 99.97 under the old 30/70 split against 99.56 +/- 0.32 under the
400-per-class protocol. The gap is real but modest, because the model is
saturated on this scene. A blanket claim that the older protocol inflates results
by a large margin would overstate it on Pavia University, whatever is true of
harder benchmarks.
