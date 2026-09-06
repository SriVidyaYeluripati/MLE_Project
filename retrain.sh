set -u
A=agent_code/model_linearQ
cost=$1
export LQ_WEIGHTS=w_bc${cost}.npz     # never touches your shipped weights.npz
export LQ_BOMB_COST=$cost
export LQ_BOMB_EXPLORE=1.0
rm -f $A/$LQ_WEIGHTS
echo "[$cost] phase 1 - 900 rounds solo"
python main.py play --no-gui --agents model_linearQ \
  --train 1 --scenario classic --n-rounds 900 > /dev/null 2>&1
echo "[$cost] phase 2 - 500 rounds vs 3x peaceful"
python main.py play --no-gui --agents model_linearQ peaceful_agent peaceful_agent peaceful_agent \
  --train 1 --scenario classic --n-rounds 500 > /dev/null 2>&1
ls -l $A/$LQ_WEIGHTS
echo "[$cost] DONE"
