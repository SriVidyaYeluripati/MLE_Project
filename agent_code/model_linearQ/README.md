# LinearQ — linear Q-learning with Expected SARSA(lambda) - A.K.A Agent GASY

Owner: Sri Vidya Yeluripati.

LinearQ scores each of the six actions with one shared weight vector `w` on
an action-relative afterstate: `features.py` computes the same 33-feature
vector for the tile the agent would occupy *after* taking an action, so one
weight applies to all four movement directions instead of learning a
separate value per direction. The model is `Q(s,a) = w · phi(s,a)`.

## Update rule

Expected SARSA(lambda), off-policy, updated online every step:

| Hyperparameter | Value | Where |
|---|---|---|
| gamma (discount) | 0.95 | `train.py`, `GAMMA` |
| lambda (eligibility trace decay) | 0.80 | `train.py`, `LAMBDA` |
| alpha (step size) | 0.01, normalized by max(‖phi(s,a)‖², 1) | `train.py`, `ALPHA` |

The eligibility trace is `gamma * lambda * trace + phi(s,a)`; the weight
update is `w += alpha_normalized * TD_error * trace`.

## Action is selected as

`callbacks.py` picks actions with a softmax over Q-values, restricted to
legal actions (an action is illegal if its `is_invalid` feature is set):

- **Training** (`self.train = True`): temperature `tau = 0.25`, so the agent
  explores in proportion to how close the Q-values are.
- **Play** (`self.train = False`): temperature `tau = 0.0` — greedy, picking
  uniformly among actions within `1e-9` of the max Q-value (tie-break).
- **Cold start**: if no weight file exists and training is starting fresh,
  the bias weight is optimistically initialized to `0.5` rather than `0`.
- **Bomb exploration**: an extra `BOMB_EXPLORE = 0.25` term biases
  exploration toward trying BOMB more than softmax alone would.

## The 33 features

Grouped as they appear in `feature_names`:

| Group | Features |
|---|---|
| Meta / action | `bias`, `is_wait`, `is_invalid` |
| Coin | `d_coin`, `coin_delta`, `no_coin` |
| Bomb availability | `is_bomb`, `bomb_ready` |
| Crates | `d_crate`, `crate_delta`, `crates_hit_1`, `crates_hit_2`, `crates_hit_3p` |
| Danger (per-step horizon) | `danger_0`, `danger_1`, `danger_2`, `danger_3`, `danger_4` |
| Safety / escape | `d_safety`, `safety_delta`, `no_escape`, `exits`, `dead_end` |
| Hand-picked interactions | `x_danger_safety`, `x_bomb_noescape`, `x_bomb_crates2` |
| Potential-based shaping | `phi_state` |
| Opponents | `opp_delta`, `opp_in_blast`, `x_bomb_opp`, `no_opp`, `opp_trapped`, `x_bomb_trapped` |

Any change to `FEATURE_NAMES` in `features.py` invalidates every saved
weight file — `callbacks.py` checks the saved feature names against the code
at load time and refuses to run on a mismatch.

## Two saved weight vectors

`weights.npz` holds two checkpoints, not one:

- `w` — the **averaged** weights (`self.w_avg` in training). This is what
  gets deployed and loaded at play time.
- `w_last` — the latest **raw, unaveraged** weights at the moment of saving.
  Kept for inspection/debugging, not loaded at play time.

The submitted file's checksum is `MD5 8ac736b30911b2f547a923fdda53ed69`.

## Files

- `features.py`   the 33-feature contract (LinearQ's own, not the shared 28-feature one used by ForestQ/GBT)
- `callbacks.py`  play: softmax/greedy action selection over the six Q-values
- `train.py`      Expected SARSA(lambda), eligibility traces, weight averaging
- `weights.npz`   trained weights — `w` (deployed, averaged), `w_last` (raw), `feature_names`, `actions`, `hyper`

## REQUIREMENTS AS STATED

    pip install numpy