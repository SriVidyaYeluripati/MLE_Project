"""
Imported only when training.   Expected SARSA(lambda) with linear features.

Why Expected SARSA rather than Q-learning: its target contains no max, so it
carries no maximisation bias at all, and it is on-policy, which is the family
that does not diverge the way Q-learning with linear function approximation can.
Both properties come free here, because act() has already computed all the
action values we need.
"""

import os
from collections import defaultdict
from typing import List

import numpy as np

import events as e

try:
    from .callbacks import action_probabilities, model_path
    from .features import (ACTIONS, N_FEATURES, FEATURE_NAMES, feature_matrix,
                           features, context, potential, describe)
except ImportError:
    from callbacks import action_probabilities, model_path
    from features import (ACTIONS, N_FEATURES, FEATURE_NAMES, feature_matrix,
                          features, context, potential, describe)

# ---------------------------------------------------------------- hyperparameters
GAMMA = 0.95        # six-step credit horizon; 1/(1-gamma) = 20
LAMBDA = 0.80       # credit assignment AND one leg of the deadly triad
ALPHA = 0.01        # normalised by ||phi||^2 below
USE_SHAPING = True  # tier 2; flip to False for the ablation row
AVG_BETA = 0.001    # iterate averaging -> the weights we actually ship
REPORT_EVERY = 100  # rounds

# ---------------------------------------------------------------- rewards
# Tier 1 - the true objective, scaled into [-1, 1] (see ch. 2 and ch. 5).
REWARDS_TRUE = {
    e.COIN_COLLECTED: 0.2,
}
# Tier 3 - declared unsafe shaping. Justify and ablate each one.
REWARDS_EXTRA = {
    e.INVALID_ACTION: -0.05,
}


def setup_training(self):
    self.trace = np.zeros(N_FEATURES)
    self.w_avg = self.w.copy()
    self.round = 0
    self.stats = defaultdict(float)
    self.history = []
    self.q_max_seen = 0.0
    self.processed_step = -1     # guards against the double delivery, see below
    self.logger.info(f'training: gamma={GAMMA} lambda={LAMBDA} alpha={ALPHA} '
                     f'shaping={USE_SHAPING}')


def reward_from(self, events, old_state, new_state, old_ctx=None):
    """tier 1 + tier 2 (potential-based) + tier 3."""
    r = sum(REWARDS_TRUE.get(ev, 0.0) for ev in events)
    self.stats['true_return'] += r
    r += sum(REWARDS_EXTRA.get(ev, 0.0) for ev in events)
    if USE_SHAPING:
        # F = gamma * Phi(s') - Phi(s).  Phi(terminal) = 0 by construction.
        r += GAMMA * potential(new_state) - potential(old_state, old_ctx)
    return r


def update(self, phi_sa, q_next, r, terminal):
    """One Expected SARSA(lambda) step."""
    q_sa = float(phi_sa @ self.w)

    if terminal:
        target = r                                   # NO bootstrap past death
    else:
        p = action_probabilities(q_next, self.tau)
        target = r + GAMMA * float(p @ q_next)

    delta = target - q_sa

    # accumulating trace, then a step size normalised by the feature norm so
    # that states with many active features do not get larger updates
    self.trace = GAMMA * LAMBDA * self.trace + phi_sa
    alpha = ALPHA / max(float(phi_sa @ phi_sa), 1.0)
    self.w += alpha * delta * self.trace
    self.w_avg += AVG_BETA * (self.w - self.w_avg)

    self.stats['td_abs'] += abs(delta)
    self.stats['updates'] += 1
    self.q_max_seen = max(self.q_max_seen, abs(q_sa))


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or self_action is None:
        return

    a = ACTIONS.index(self_action) if self_action in ACTIONS else None
    if a is None:                       # framework substituted WAIT on timeout
        return

    old_ctx = getattr(self, 'last_ctx', None)
    phi_sa = (self.last_phi[a] if getattr(self, 'last_phi', None) is not None
              else features(old_game_state, self_action))
    q_next = feature_matrix(new_game_state) @ self.w

    r = reward_from(self, events, old_game_state, new_game_state, old_ctx)
    update(self, phi_sa, q_next, r, terminal=False)

    for ev in events:
        self.stats[ev] += 1
    self.processed_step = old_game_state['step']


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Two different situations arrive here, and treating them alike corrupts the
    single most important transition in the game.

    * We DIED.  environment.send_game_events() skips dead agents, so this is the
      only delivery of that step.  It is a true terminal: no bootstrap.
    * We SURVIVED to the step limit.  send_game_events() already delivered this
      exact step to game_events_occurred, and end_round() re-sends the same
      (unreset) event list with SURVIVED_ROUND appended.  Updating again would
      apply the same transition twice and double-count its reward.  It is also
      not a terminal - the episode was truncated by the clock - so the bootstrap
      already applied in game_events_occurred is the correct treatment.

    self.processed_step tells the two apart.
    """
    already_seen = (last_game_state is not None
                    and last_game_state['step'] == self.processed_step)

    if not already_seen and last_action in ACTIONS:
        a = ACTIONS.index(last_action)
        phi_sa = (self.last_phi[a] if getattr(self, 'last_phi', None) is not None
                  else features(last_game_state, last_action))
        r = reward_from(self, events, last_game_state, None,
                        getattr(self, 'last_ctx', None))
        update(self, phi_sa, None, r, terminal=True)

    for ev in events:
        if not already_seen or ev == e.SURVIVED_ROUND:   # the only genuinely new one
            self.stats[ev] += 1

    self.trace[:] = 0.0
    self.last_phi = self.last_q = self.last_ctx = None
    self.processed_step = -1
    self.round += 1

    # last_game_state predates the final step's collection, so count the score
    # from events instead of reading a stale field
    self.stats['score'] = (1.0 * self.stats[e.COIN_COLLECTED]
                           + 5.0 * self.stats[e.KILLED_OPPONENT])
    self.stats['steps'] += last_game_state['step']

    if self.round % REPORT_EVERY == 0:
        n = REPORT_EVERY
        row = {
            'round': self.round,
            'score': self.stats['score'] / n,
            'coins': self.stats[e.COIN_COLLECTED] / n,
            'survived': self.stats[e.SURVIVED_ROUND] / n,
            'invalid': self.stats[e.INVALID_ACTION] / n,
            'steps': self.stats['steps'] / n,
            'td_abs': self.stats['td_abs'] / max(self.stats['updates'], 1),
            'w_norm': float(np.linalg.norm(self.w)),
            'max_abs_q': self.q_max_seen,
        }
        self.history.append(row)
        self.logger.info(str(row))
        print(f"[{self.round:5d}]  score {row['score']:5.2f}  coins {row['coins']:5.2f}"
              f"  invalid {row['invalid']:5.2f}  |w| {row['w_norm']:5.2f}"
              f"  max|Q| {row['max_abs_q']:5.2f}  |d| {row['td_abs']:.4f}")
        print('          ' + describe(self.w))
        self.stats = defaultdict(float)
        self.q_max_seen = 0.0
        save(self)


def save(self):
    np.savez(model_path(),
             w=self.w_avg,                 # ship the AVERAGE, not the last iterate
             w_last=self.w,
             feature_names=np.array(FEATURE_NAMES),
             actions=np.array(ACTIONS),
             hyper=np.array([GAMMA, LAMBDA, ALPHA, float(USE_SHAPING)]))