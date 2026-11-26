#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/c10_region_restore_20261029_v1
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export YOLO_CONFIG_DIR="$ROOT/cache/yolo"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
export WANDB_MODE=disabled
export COMET_MODE=DISABLED
cd "$ROOT"
test -f receipts/offline_setup_exit_code.txt
test -f receipts/data_preparation.json
test -f receipts/evaluator_fixture_server.json
if test -f receipts/queue.pid && kill -0 "$(cat receipts/queue.pid)" 2>/dev/null; then
    printf 'An existing queue is running; refusing duplicate launch.\n'
    exit 1
fi
nohup "$ROOT/venv/bin/python" code/run_queue.py > logs/queue.log 2>&1 < /dev/null &
printf '%s\n' "$!" > receipts/queue.pid
ps -p "$!" -o pid,etime,args
