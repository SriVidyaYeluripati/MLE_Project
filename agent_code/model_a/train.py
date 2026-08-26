"""
Model A - train.py  (the part that LEARNS)

Fitted Q-Iteration: keep one regressor per action. Every REFIT_EVERY rounds,
rebuild the Q-learning targets  r + GAMMA * max_a' Q(s', a')  and refit all
six regressors on the accumulated transitions.

------------------------------ HOW TO EXPERIMENT ------------------------------
Every tunable is a named constant below. Change ONE at a time, run a fixed
evaluation, and log the before/after number. The report is graded on exactly
this kind of controlled comparison, so a change you didn't measure is a
result you can't report.
-------------------------------------------------------------------------------
"""
import pickle
from collections import deque
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import events as e
from .callbacks import ACTIONS, MODEL_FILE
from .features import state_to_features, N_FEATURES

# ---- hyperparameters (change ONE at a time, log the result) -------------
GAMMA            = 0.9       # discount factor
BUFFER_SIZE      = 100_000   # transitions kept in memory
REFIT_EVERY      = 20        # refit the regressors every N rounds
N_TREES          = 40        # forest size (forest regressor only)
MIN_SAMPLES      = 500       # don't fit before we have this many transitions
EPS_START        = 1.0       # exploration at the start
EPS_END          = 0.05      # exploration floor
EPS_DECAY_ROUNDS = 400       # rounds over which epsilon decays
REGRESSOR        = "forest"  # "forest" or "ridge"  <- a free experiment
USE_SYMMETRY     = True      # 8x data augmentation from board symmetries
# -------------------------------------------------------------------------

# The board has 8 symmetries (4 rotations x mirror). Under a symmetry the
# direction-based features permute, and so does the action. Every group of 4
# URDL features starts at one of these indices in FEATURE_NAMES:
DIR_BLOCKS = [0, 4, 8, 12, 17, 24]   # free, coin, crate, danger, escape, enemy


def _permute(features, action_idx, rot, mirror):
    """Apply one board symmetry to a feature vector and its action index."""
    f = features.copy()
    perm = [(i - rot) % 4 for i in range(4)]        # rotate URDL slots
    for s in DIR_BLOCKS:
        block = f[s:s + 4][perm]
        if mirror:
            block = block[[0, 3, 2, 1]]             # mirror swaps R and L
        f[s:s + 4] = block
    if action_idx < 4:                              # a move action rotates too
        a = (action_idx - rot) % 4
        if mirror:
            a = [0, 3, 2, 1][a]
    else:
        a = action_idx                              # WAIT / BOMB are unchanged
    return f, a


def setup_training(self):
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.round_count = 0
    self.epsilon = EPS_START
    self.logger.info(f"Model A training: {N_FEATURES} features, {REGRESSOR}, "
                     f"symmetry={USE_SYMMETRY}.")


def _record(self, old_state, action, new_state, reward):
    if old_state is None or action not in ACTIONS:
        return
    s  = state_to_features(old_state)
    s2 = state_to_features(new_state) if new_state is not None else None
    a  = ACTIONS.index(action)
    variants = [(r, m) for r in range(4) for m in (False, True)] if USE_SYMMETRY \
        else [(0, False)]
    for rot, mirror in variants:
        fs, fa = _permute(s, a, rot, mirror)
        fs2 = _permute(s2, a, rot, mirror)[0] if s2 is not None else None
        self.buffer.append((fs, fa, reward, fs2))


def game_events_occurred(self, old_game_state, self_action,
                         new_game_state, events):
    _record(self, old_game_state, self_action,
            new_game_state, reward_from_events(self, events))


def end_of_round(self, last_game_state, last_action, events):
    _record(self, last_game_state, last_action,
            None, reward_from_events(self, events))

    self.round_count += 1
    frac = min(1.0, self.round_count / EPS_DECAY_ROUNDS)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    if self.round_count % REFIT_EVERY == 0 and len(self.buffer) >= MIN_SAMPLES:
        _refit(self)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"Refit @ round {self.round_count}: "
                         f"{len(self.buffer)} transitions, eps={self.epsilon:.3f}")


def _refit(self):
    """One step of fitted Q-iteration: build targets, refit six regressors."""
    S  = np.array([t[0] for t in self.buffer])
    A  = np.array([t[1] for t in self.buffer])
    R  = np.array([t[2] for t in self.buffer], dtype=float)
    S2 = [t[3] for t in self.buffer]

    # bootstrap term: max_a' Q(s', a') using the CURRENT model
    future = np.zeros(len(R))
    nonterminal = np.array([s is not None for s in S2])
    if self.model is not None and nonterminal.any():
        X2 = np.array([s for s in S2 if s is not None])
        qs = np.column_stack([reg.predict(X2) for reg in self.model])
        future[nonterminal] = qs.max(axis=1)

    Y = R + GAMMA * future

    new_model = []
    for a in range(len(ACTIONS)):
        mask = A == a
        if mask.sum() < 10:                       # too little data this action
            reg = _fresh_regressor().fit(np.zeros((1, N_FEATURES)), [0.0])
        else:
            reg = _fresh_regressor().fit(S[mask], Y[mask])
        new_model.append(reg)
    self.model = new_model


def _fresh_regressor():
    if REGRESSOR == "ridge":
        return Ridge(alpha=1.0)
    return RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=5,
                                 n_jobs=-1, random_state=0)


def reward_from_events(self, events: List[str]) -> float:
    """Reward table. Change ONE value at a time and log the result."""
    rewards = {
        e.COIN_COLLECTED:   3.0,
        e.KILLED_OPPONENT:  5.0,
        e.CRATE_DESTROYED:  0.3,
        e.COIN_FOUND:       0.2,
        e.INVALID_ACTION:  -1.0,
        e.KILLED_SELF:     -5.0,
        e.GOT_KILLED:      -5.0,
        e.WAITED:          -0.05,
    }
    total = sum(rewards.get(ev, 0.0) for ev in events)
    total -= 0.02                      # small per-step cost discourages stalling
    return total
