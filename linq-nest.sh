#!/usr/bin/env bash
# Re-nest the LinQ work so that every top-level folder keeps its upstream name
# and this agent's content lives in a model_linq/ subfolder inside it:
#
#     results/model_linq/...        logs/model_linq/...
#     experiments/model_linq/...    tools/model_linq/...
#
# A teammate adding results/model_gbt/ or logs/model_dqn/ can never collide.
# agent_code/model_linearQ/ is untouched - the framework requires agents there.
#
# Run from the repo root:   bash linq-nest.sh
set -eu
[ -f main.py ] || { echo "run this from the repo root (where main.py is)"; exit 1; }
[ -d linq ]    || { echo "no linq/ directory - nothing to re-nest"; exit 1; }

M=model_linq
mkdir -p results/$M logs/$M experiments/$M tools/$M

mv_into() {   # $1 = source path, $2 = destination directory
  [ -e "$1" ] || return 0
  git mv "$1" "$2/" 2>/dev/null || mv "$1" "$2/"
  echo "  $1 -> $2/"
}

# data and documents
if [ -d linq/results ]; then
  for x in linq/results/*; do mv_into "$x" "results/$M"; done
  rmdir linq/results 2>/dev/null || true
fi
if [ -d linq/experiments ]; then
  for x in linq/experiments/*; do mv_into "$x" "experiments/$M"; done
  rmdir linq/experiments 2>/dev/null || true
fi
if [ -d linq/logs ]; then
  for x in linq/logs/*; do mv_into "$x" "logs/$M"; done
  rmdir linq/logs 2>/dev/null || true
fi
[ -d linq/.exp_done ]   && { mv linq/.exp_done   "results/$M/.done"      2>/dev/null || true; }
[ -d linq/.trapc_done ] && { mv linq/.trapc_done "results/$M/.trapc_done" 2>/dev/null || true; }
[ -d .exp_done ]        && { mv .exp_done        "results/$M/.done"      2>/dev/null || true; }
[ -d .trapc_done ]      && { mv .trapc_done      "results/$M/.trapc_done" 2>/dev/null || true; }

# everything still in linq/ is tooling
for x in linq/*; do mv_into "$x" "tools/$M"; done
rmdir linq 2>/dev/null || true

echo "repointing the scripts (now two levels down)..."
R='tools/'"$M"
sed -i 's|/\.\." && pwd)"|/../.." \&\& pwd)"|' "$R/run_experiments.sh" "$R/trapc_run.sh" 2>/dev/null || true
sed -i 's|linq/results/|results/'"$M"'/|g; s|linq/\.exp_done|results/'"$M"'/.done|g; s|linq/\.trapc_done|results/'"$M"'/.trapc_done|g' \
       "$R/run_experiments.sh" "$R/trapc_run.sh" 2>/dev/null || true
sed -i 's|linq/experiments/|experiments/'"$M"'/|g; s|linq/exp_report\.py|tools/'"$M"'/exp_report.py|g' \
       "$R/run_experiments.sh" 2>/dev/null || true
sed -i 's|linq/results/|results/'"$M"'/|g' "$R/exp_report.py" "$R/trapc.py" 2>/dev/null || true
# the herding probe is imported by callbacks via a relative path
sed -i "s|'\.\.', '\.\.', 'experiments'|'..', '..', '..', 'experiments', '$M'|" \
       agent_code/model_linearQ/callbacks.py 2>/dev/null || true

bash -n "$R/run_experiments.sh" && python3 -c "compile(open('$R/exp_report.py').read(),'x','exec')"
python3 -m py_compile agent_code/model_linearQ/*.py
echo
echo "root:"; ls -p | grep /
echo "results/:"; ls results/ 2>/dev/null
echo "done."
