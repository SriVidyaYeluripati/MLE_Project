#!/usr/bin/env bash
# Runs the two pre-registered experiments, in order, resumably.
#
#   bash run_experiments.sh gamma     PREREG_gamma.md    3 arms x 12 seeds
#   bash run_experiments.sh herd      PREREG_herding.md  2 arms x 40 seeds
#   bash run_experiments.sh both
#
# Uses every core.  Override with  JOBS=8 bash run_experiments.sh both
# Safe to re-run after any interruption: a step is skipped only when its
# marker AND its output file are both present.
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"
export A=agent_code/model_linearQ
JOBS="${JOBS:-$(nproc)}"
PHASE_SEED_BASE=5000
case "${1:-both}" in phase) PHASE_SEED_BASE=7000 ;; esac
mkdir -p results/model_linq/.done results/model_linq/gamma results/model_linq/herd results/model_linq/phase logs

train_one() {                       # "<tag> <seed> <exp>"
  set -- $1; local tag=$1 s=$2 exp=$3 sw=""
  case "$tag" in g80) sw="LQ_GAMMA=0.80";; g60) sw="LQ_GAMMA=0.60";;
                 hherd) sw="LQ_HERD=1";; pphase) sw="LQ_PHASE=1";; esac
  local m=results/model_linq/.done/train_$tag$s
  [ -f "$m" ] && [ -f "$A/w_$tag$s.npz" ] && { echo "skip  train $tag$s"; return; }
  rm -f "$A/w_$tag$s.npz"; cp "$A/weights.npz" "$A/w_$tag$s.npz"
  env LQ_WEIGHTS=w_$tag$s.npz LQ_BOMB_COST=0 LQ_CRATE_VALUE=0.02 $sw \
    python main.py play --no-gui \
    --agents model_linearQ peaceful_agent peaceful_agent peaceful_agent \
    --train 1 --scenario classic --n-rounds 600 --seed $((PHASE_SEED_BASE+s)) >/dev/null 2>&1 \
    && touch "$m" && echo "train $tag$s done $(date +%H:%M)"
}
eval_one() {                        # "<tag> <seed> <exp>"
  set -- $1; local tag=$1 s=$2 exp=$3 sw=""
  case "$tag" in g80) sw="LQ_GAMMA=0.80";; g60) sw="LQ_GAMMA=0.60";;
                 hherd) sw="LQ_HERD=1";; pphase) sw="LQ_PHASE=1";; esac
  local m=results/model_linq/.done/eval_$tag$s
  [ -f "$m" ] && [ -s "results/model_linq/$exp/$tag$s.json" ] && { echo "skip  eval $tag$s"; return; }
  local opp=coin_collector_agent
  [ "$exp" = phase ] && opp=rule_based_agent      # see PREREG_phase.md
  env LQ_WEIGHTS=w_$tag$s.npz $sw python main.py play --no-gui \
    --agents model_linearQ $opp $opp $opp \
    --scenario classic --n-rounds 300 --save-stats results/model_linq/$exp/$tag$s.json >/dev/null 2>&1 \
    && touch "$m" && echo "eval  $tag$s done $(date +%H:%M)"
}
export -f train_one eval_one

jobs_gamma(){ for s in $(seq 1 12);  do for t in g95 g80 g60; do echo "$t $s gamma"; done; done; }
jobs_herd(){  for s in $(seq 1 40);  do for t in hbase hherd;  do echo "$t $s herd";  done; done; }
jobs_phase(){ for s in $(seq 1 40);  do for t in pbase pphase; do echo "$t $s phase"; done; done; }

phase(){ local name=$1 lister=$2 n
  n=$($lister | wc -l)
  echo "=== $name: $n trainings, $JOBS parallel on $(nproc) cores ==="
  $lister | xargs -P "$JOBS" -I{} bash -c 'train_one "$@"' _ {}
  echo "=== $name: $n evaluations ==="
  $lister | xargs -P "$JOBS" -I{} bash -c 'eval_one "$@"' _ {}
}

case "${1:-both}" in
  gamma) phase GAMMA jobs_gamma; python3 tools/model_linq/exp_report.py gamma ;;
  herd)  phase HERD  jobs_herd;  python3 tools/model_linq/exp_report.py herd ;;
  phase) phase PHASE jobs_phase; python3 tools/model_linq/exp_report.py phase ;;
  both)  phase GAMMA jobs_gamma; python3 tools/model_linq/exp_report.py gamma
         phase HERD  jobs_herd;  python3 tools/model_linq/exp_report.py herd ;;
  *) echo "usage: bash run_experiments.sh [gamma|herd|phase|both]"; exit 1 ;;
esac
