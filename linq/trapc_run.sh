#!/usr/bin/env bash
# Confirmatory test of the LQ_FIX_TRAPPED coins effect.
# Pre-registered in experiments/PREREG_trapped_confirm.md BEFORE any run.
#   coins is PRIMARY, one-sided, t > 2.262 at n = 10.
#
# Resumable: every finished step writes a marker in linq/.trapc_done/.
# Re-running after an interruption skips what is already complete and
# redoes only the step that was cut off.  Safe to run as many times as needed.
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R"
A=agent_code/model_linearQ
mkdir -p linq/.trapc_done linq/results/trapc logs

train() {                                    # $1 tag  $2 seed-index  $3 env
  local m=linq/.trapc_done/train_$1$2
  [ -f "$m" ] && [ -f "$A/w_$1$2.npz" ] && { echo "skip  train $1$2"; return; }
  rm -f $A/w_$1$2.npz
  cp $A/weights.npz $A/w_$1$2.npz
  env LQ_WEIGHTS=w_$1$2.npz LQ_BOMB_COST=0 LQ_CRATE_VALUE=0.02 $3 \
    python main.py play --no-gui \
    --agents model_linearQ peaceful_agent peaceful_agent peaceful_agent \
    --train 1 --scenario classic --n-rounds 600 --seed $((4000+$2)) >/dev/null 2>&1 \
    && touch "$m" && echo "train $1$2  done  $(date +%H:%M)"
}

ev() {                                       # $1 tag+seed  $2 env
  local m=linq/.trapc_done/eval_$1
  [ -f "$m" ] && [ -s "linq/results/trapc/$1.json" ] && { echo "skip  eval $1"; return; }
  env LQ_WEIGHTS=w_$1.npz $2 python main.py play --no-gui \
    --agents model_linearQ coin_collector_agent coin_collector_agent coin_collector_agent \
    --scenario classic --n-rounds 300 --save-stats linq/results/trapc/$1.json >/dev/null 2>&1 \
    && touch "$m" && echo "eval  $1  done  $(date +%H:%M)"
}

lane() { for s in $1; do train cbase $s "X=1"; train cfix $s "LQ_FIX_TRAPPED=1"; done; }
elane(){ for s in $1; do ev cbase$s "X=1";     ev cfix$s "LQ_FIX_TRAPPED=1";     done; }

echo "=== training (20 runs, ~12 min each, 2 lanes) ==="
lane "1 2 3 4 5"  &
lane "6 7 8 9 10" &
wait
echo "=== evaluating (20 runs, ~2 min each) ==="
elane "1 2 3 4 5"  &
elane "6 7 8 9 10" &
wait
echo
echo "=== RESULT ==="
python3 trapc.py
