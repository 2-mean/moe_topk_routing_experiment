#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/moe_topk_routing_experiment"
source "$HOME/miniconda3/bin/activate" cas4160
export PYTHONPATH="$PWD/src"
WAIT_SESSION="matryo_topk__skip_wait__" bash scripts/run_mechanism_overnight_queue.sh 2>&1 | tee /tmp/2020110906_matryo_topk/logs/mechanism_overnight_queue_nowait.log
