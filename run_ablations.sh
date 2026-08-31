#!/usr/bin/env bash
# Step 5 - the ablation suite for Model LinQ.
# Each variant is trained FROM SCRATCH and evaluated, REPS times.  The repeats
# are not decoration: the full model measured four times gave margins of +1.53,
# +1.32, +0.92 and +0.50 from identical code, because training lands on either
# a cautious or an aggressive policy.  One run per variant cannot separate an
# ablation from that.
#
#   bash run_ablations.sh                     # default variant list
#   bash run_ablations.sh no_escape no_opp    # only these
#   REPS=5 bash run_ablations.sh no_escape    # more repeats
set -u
cd "$(dirname "$0")"

REPS=${REPS:-3}
TRAIN_SOLO=400
TRAIN_PEACE=300
EVAL=200

mkdir -p results/ablation

VARIANTS=(
  "full         :"
  "no_escape    :LQ_ABLATE=escape"
  "no_opp       :LQ_ABLATE=opp"
  "no_conj      :LQ_ABLATE=conj"
  "no_shaping   :LQ_SHAPING=0"
  "no_mask      :LQ_MASK_INVALID=0"
)

WANTED=("$@")
want() {
  [ ${#WANTED[@]} -eq 0 ] && return 0
  for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

for w in "${WANTED[@]}"; do
  found=0
  for entry in "${VARIANTS[@]}"; do
    [ "$(echo "${entry%%:*}" | tr -d ' ')" = "$w" ] && found=1
  done
  [ $found -eq 0 ] && VARIANTS+=("$w:LQ_ABLATE=${w#no_}")
done

for entry in "${VARIANTS[@]}"; do
  name="$(echo "${entry%%:*}" | tr -d ' ')"
  envs="${entry#*:}"
  want "$name" || continue

  for rep in $(seq 1 "$REPS"); do
    stats="results/ablation/${name}__r${rep}.json"
    if [ -f "$stats" ]; then echo "== $name rep $rep: done, skipping"; continue; fi

    wfile="w_${name}_r${rep}.npz"
    rm -f "agent_code/model_linearQ/${wfile}"
    export LQ_WEIGHTS="$wfile"
    unset LQ_ABLATE LQ_SHAPING LQ_LAMBDA LQ_MASK_INVALID
    [ -n "$envs" ] && export ${envs}

    echo "== $name rep $rep  (${envs:-no overrides})  ->  $stats"
    python main.py play --no-gui --agents model_linearQ \
        --train 1 --scenario classic --n-rounds $TRAIN_SOLO > /dev/null 2>&1
    python main.py play --no-gui \
        --agents model_linearQ peaceful_agent peaceful_agent peaceful_agent \
        --train 1 --scenario classic --n-rounds $TRAIN_PEACE > /dev/null 2>&1
    python main.py play --no-gui \
        --agents model_linearQ rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --n-rounds $EVAL --save-stats "$stats" > /dev/null 2>&1
    rm -f "agent_code/model_linearQ/${wfile}"
  done
done

unset LQ_WEIGHTS LQ_ABLATE LQ_SHAPING LQ_LAMBDA LQ_MASK_INVALID
echo
python collect_ablations.py
