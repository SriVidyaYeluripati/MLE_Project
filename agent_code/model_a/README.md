# Model A — Fitted Q-Iteration (regression forests / ridge)

Our second model, using **lecture techniques** (regression forests from the
tree-methods exercise, ridge from ex05) instead of a neural network. It is the
always-submittable fallback and satisfies the project's "at least one model
using lecture techniques" requirement.

## Files
- `features.py` — **shared team contract.** 28 features. Model B (the DQN)
  should import from here too, so both models run on identical input.
- `callbacks.py` — the playing part: `act()` asks six regressors for Q-values
  and takes the best.
- `train.py` — the learning part: fitted Q-iteration with 8× symmetry
  augmentation and epsilon decay.

## Install
```
pip install numpy scikit-learn
```
(These files go in `bomberman_rl/agent_code/model_a/`.)

## Train
```
python main.py play --no-gui --agents model_a --scenario coin-heaven --train 1 --n-rounds 300
```
Then progress to `loot-crate`, then `classic`. The saved model is `model_a.pt`.

## Play / watch
```
python main.py play --agents model_a --scenario coin-heaven
```

## Verified result
On coin-heaven (50 eval rounds), a trained Model A collected **50/50 coins per
round with 0 suicides — matching the hand-coded `coin_collector_agent`**, while
`random_agent` gets ~1.7 coins and suicides every round.

## How it works (one paragraph)
We keep six regressors, one per action. Each predicts Q(s, a) = "how good is
this action in this state?" To act, we ask all six and take the highest. To
learn, every 20 rounds we rebuild the target `r + gamma * max_a' Q(s', a')` for
every stored transition and refit the regressors on it. That's fitted
Q-iteration — value-based RL using the regressors from the course.

## Experiments already wired up (change ONE, measure, log)
Top of `train.py`:
- `REGRESSOR = "forest"` vs `"ridge"` — forest vs linear
- `USE_SYMMETRY = True` vs `False` — does 8× augmentation help sample efficiency
- `GAMMA`, `EPS_DECAY_ROUNDS`, `REFIT_EVERY`, `N_TREES`
- the reward table in `reward_from_events`

Each of these is a ready-made controlled experiment for the report.

## Shared-contract note for Shayan (Model B)
```python
from .features import state_to_features, N_FEATURES
N_OBSERVATIONS = N_FEATURES   # = 28, replaces the old 5
```
Then the DQN sees coins, danger, and escape routes — the missing information
that was causing the 20/20 suicides.
