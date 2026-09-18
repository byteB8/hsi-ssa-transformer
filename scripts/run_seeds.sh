#!/usr/bin/env bash
# Repeat the fixed-protocol configurations across seeds.
#
# Single-seed differences on Pavia University are not trustworthy: every model
# here scores above 99% OA, so gaps of a few tenths sit inside run-to-run
# variation. Dang et al. average 10 repeats for this reason.
set -euo pipefail

PYTHON="${PYTHON:-$HOME/env/ml/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p reports/seeds logs/seeds

EPOCHS="${EPOCHS:-80}"
SEEDS="${SEEDS:-1 2}"

run () {
  local name="$1" seed="$2"; shift 2
  local tag="${name}_s${seed}"
  [ -f "reports/seeds/${tag}.json" ] && { echo "skip $tag"; return; }
  echo "=== $tag ==="
  "$PYTHON" -m hsi_ssat.train --epochs "$EPOCHS" --seed "$seed" \
    --out "reports/seeds/${tag}.json" "$@" > "logs/seeds/${tag}.log" 2>&1
  tail -n 2 "logs/seeds/${tag}.log"
}

for seed in $SEEDS; do
  case "${GROUP:-all}" in
    cnn)
      run hybridsn      "$seed" --model hybridsn      --window 25 --pca-components 15 --protocol fixed --per-class 400
      run hybridsn_cbam "$seed" --model hybridsn_cbam --window 25 --pca-components 15 --protocol fixed --per-class 400
      ;;
    transformer)
      run ssat          "$seed" --model ssa_transformer         --window 15 --pca-components 0 --protocol fixed --per-class 400
      run ssat_nodense  "$seed" --model ssa_transformer_nodense --window 15 --pca-components 0 --protocol fixed --per-class 400
      ;;
  esac
done
