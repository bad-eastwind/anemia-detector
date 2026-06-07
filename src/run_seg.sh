#!/bin/zsh
# Generic segmentation runner. Usage:  zsh src/run_seg.sh <config.yaml>
# Runs every mode listed inside the config (cv / loco / final), logging to outputs/seg/logs/.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python              # on HPC use the active env's python
export TOKENIZERS_PARALLELISM=false
CFG=${1:?usage: run_seg.sh <config.yaml>}
NAME=$(basename "$CFG" .yaml)
mkdir -p outputs/seg/logs
echo "=== SEG $NAME start $(date) ==="
$PY src/seg.py --config "$CFG" 2>&1 | tee "outputs/seg/logs/${NAME}.log"
echo "=== SEG $NAME done  $(date) ==="
