#!/usr/bin/env bash
# ===========================================================================
# The two action filters of section 8.  Play-time only - no retraining, the
# weights are identical in every arm.  Both filters default OFF; this script
# is the experiment that says why they stay off.
#
#     bash experiments/run_action_filters.sh measure   # 4 runs x 1000 rounds
#     bash experiments/run_action_filters.sh probe     # does the filter fire?
#     bash experiments/run_action_filters.sh report    # print the table
#
# Run the probe FIRST if you are ever tempted to believe a margin difference
# here: the safety filter changes the chosen action 0.02% of the time, which
# is why its apparently clean +0.21 was noise (finding 16).
# ===========================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p results/safefilter

if [ "${1:-}" = "measure" ]; then
  for opp in rule_based_agent coin_collector_agent; do
    tag=$([ "$opp" = rule_based_agent ] && echo rb || echo cc)
    for cfg in "off 0 0" "safe 1 0" "bomb 0 1" "both 1 1"; do
      set -- $cfg
      echo "running $tag/$1 ..."
      LQ_SAFE_FILTER=$2 LQ_BOMB_FILTER=$3 python main.py play --no-gui \
        --agents model_linearQ $opp $opp $opp --scenario classic \
        --n-rounds 1000 --save-stats results/safefilter/${tag}_${1}.json \
        >/dev/null 2>&1
    done
  done
  exit 0
fi

if [ "${1:-}" = "probe" ]; then
  for opp in rule_based_agent coin_collector_agent; do
    echo "=== filter-bite rate vs $opp ==="
    LQ_SAFE_FILTER=1 LQ_BOMB_FILTER=0 LQ_PROBE=1 python main.py play --no-gui \
      --agents model_linearQ $opp $opp $opp --scenario classic \
      --n-rounds 250 2>&1 | grep -o 'PROBE steps=.*' | tail -2
  done
  exit 0
fi

if [ "${1:-}" = "report" ]; then
  python - <<'PY'
import json, os
def row(tag):
    p = f'results/safefilter/{tag}.json'
    if not os.path.exists(p):
        return
    d = json.load(open(p))['by_agent']; me = 'model_linearQ'
    best = max(v['score']/v['rounds'] for k, v in d.items() if k != me)
    r = d[me]; n = r['rounds']; g = lambda k: r.get(k, 0)/n
    print(f"  {tag:10s} n={n:5d}  margin {g('score')-best:+6.2f}  score {g('score'):5.2f}  "
          f"coins {g('coins'):5.2f}  crates {g('crates'):6.2f}  bombs {g('bombs'):6.2f}  "
          f"cr/bomb {g('crates')/max(g('bombs'), 1e-9):4.2f}  suic {g('suicides'):4.2f}")
for opp, label in (('rb', '3x rule_based'), ('cc', '3x coin_collector')):
    print(f"\nvs {label}")
    for cfg in ('off', 'safe', 'bomb', 'both'):
        row(f'{opp}_{cfg}')
PY
  exit 0
fi

echo "usage: bash experiments/run_action_filters.sh {measure|probe|report}"
