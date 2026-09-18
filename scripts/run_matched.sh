#!/usr/bin/env bash
# Architecture comparison at matched input geometry.
#
# The main sweep gives each model its native input: HybridSN sees 25x25 patches
# over 15 PCA bands, the SSA-Transformer sees 15x15 over all 103. Any difference
# between them therefore confounds architecture with spatial context and with
# spectral preprocessing.
#
# Here both models get identical input. Two settings are used so the conclusion
# does not rest on a single choice of geometry:
#
#   m15_pca15   15x15 patches, 15 PCA bands   (compact)
#   m15_full    15x15 patches, all 103 bands  (no spectral reduction)
#
# 15 is the largest window the transformer accepts below HybridSN's native 25
# while staying divisible by its patch size of 3. Optimiser, schedule, batch
# size and seeds are identical throughout.
set -euo pipefail

PYTHON="${PYTHON:-$HOME/env/ml/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
mkdir -p reports/matched logs/matched

EPOCHS="${EPOCHS:-80}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"

run () {
  local model="$1" setting="$2" pca="$3" seed="$4"
  local tag="${model}_${setting}_s${seed}"
  [ -f "reports/matched/${tag}.json" ] && { echo "skip $tag"; return; }
  "$PYTHON" -m hsi_ssat.train --epochs "$EPOCHS" --seed "$seed" \
    --model "$model" --window 15 --pca-components "$pca" \
    --protocol fixed --per-class 400 \
    --out "reports/matched/${tag}.json" > "logs/matched/${tag}.log" 2>&1
  printf '%-34s %s\n' "$tag" "$(grep -E '^(hybridsn|ssa_transformer)' "logs/matched/${tag}.log" | tail -1)"
}

for seed in $SEEDS; do
  case "${GROUP:-all}" in
    cnn)
      run hybridsn        m15_pca15 15 "$seed"
      run hybridsn        m15_full   0 "$seed"
      ;;
    transformer)
      run ssa_transformer m15_pca15 15 "$seed"
      run ssa_transformer m15_full   0 "$seed"
      ;;
  esac
done
