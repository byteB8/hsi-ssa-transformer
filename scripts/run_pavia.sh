#!/usr/bin/env bash
# Pavia University benchmark sweep.
#
# All runs share one split protocol (Dang et al.: 400 labelled pixels per class,
# the rest test) and one seed, so the numbers are comparable. Each architecture
# otherwise keeps its own native input geometry: HybridSN wants 25x25 patches
# over 15 PCA bands, the SSA-Transformer wants 15x15 patches over all 103 bands.
#
# The final run deliberately uses the older 30/70 random split to show how much
# that protocol inflates accuracy on this scene.
set -euo pipefail

PYTHON="${PYTHON:-$HOME/env/ml/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p reports logs

EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-0}"

run () {
  local tag="$1"; shift
  echo "=== $tag ==="
  "$PYTHON" -m hsi_ssat.train --epochs "$EPOCHS" --seed "$SEED" \
    --out "reports/${tag}.json" "$@" 2>&1 | tee "logs/${tag}.log"
}

case "${GROUP:-all}" in
  cnn)
    run hybridsn        --model hybridsn      --window 25 --pca-components 15 --protocol fixed --per-class 400
    run hybridsn_cbam   --model hybridsn_cbam --window 25 --pca-components 15 --protocol fixed --per-class 400
    run hybridsn_ratio  --model hybridsn      --window 25 --pca-components 15 --protocol ratio --test-ratio 0.7
    ;;
  transformer)
    run ssat            --model ssa_transformer         --window 15 --pca-components 0 --protocol fixed --per-class 400
    run ssat_nodense    --model ssa_transformer_nodense --window 15 --pca-components 0 --protocol fixed --per-class 400
    ;;
  *)
    echo "set GROUP=cnn or GROUP=transformer" >&2; exit 1
    ;;
esac
