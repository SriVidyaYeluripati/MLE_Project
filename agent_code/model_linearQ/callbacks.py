"""
Always loaded, including in the tournament.   Keep this file fast and free of
any training-only imports: train.py does not exist at tournament time.
"""

import os

import numpy as np

try:
    from .features import (ACTIONS, N_FEATURES, BIAS, NO_ESCAPE,
                           feature_matrix, context)
except ImportError:
    from features import (ACTIONS, N_FEATURES, BIAS, NO_ESCAPE,
                          feature_matrix, context)

MODEL_FILE = 'weights.npz'          # relative to THIS file - never absolute

OPTIMISTIC_INIT = 0.5               # untried actions look attractive (ch. 4)
TAU_TRAIN = 0.25                    # softmax temperature while training
TAU_PLAY = 0.0                      # near-greedy when it counts
TIE_EPS = 1e-9                      # values this close count as tied

# Exploration is NOT uniform over the six actions: a random BOMB can end the
# episode, and an episode that ends early stops producing data (ch. 4.2).
BOMB_EXPLORE = 0.25                 # BOMB is sampled at a quarter of its softmax share
MASK_FATAL_ROUNDS = 0               # curriculum: mask provably-fatal actions for the
                                    # first N training rounds, then learn from real
                                    # deaths. 0 = off. Ablate this.


def model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)


def setup(self):
    """Called once before the first round."""
    path = model_path()
    self.rng = np.random.default_rng()

    if os.path.isfile(path):
        data = np.load(path, allow_pickle=True)
        self.w = data['w'].astype(float)
        self.logger.info(f'loaded weights from {MODEL_FILE}')
    elif getattr(self, 'train', False):
        self.w = np.zeros(N_FEATURES)
        self.w[BIAS] = OPTIMISTIC_INIT
        self.logger.info('no weights found - starting from optimistic init')
    else:
        raise FileNotFoundError(
            f'{MODEL_FILE} is missing; train the agent before playing')

    self.tau = TAU_TRAIN if getattr(self, 'train', False) else TAU_PLAY


def action_probabilities(q, tau, rng=None):
    """
    Scale-aware softmax.  A fixed temperature is NOT scale free: early in
    training all Q-values sit near zero (near-uniform) and late they are far
    apart (near-greedy), so the exploration rate would anneal itself in a way
    we never chose.  Dividing by the spread of the values fixes that.
    """
    q = np.asarray(q, dtype=float)
    if tau <= 0:
        p = (q >= q.max() - TIE_EPS).astype(float)   # greedy, ties shared
        return p / p.sum()
    spread = q.std()
    if spread < 1e-8:
        return np.full(len(q), 1.0 / len(q))         # all tied
    z = (q - q.max()) / (tau * spread)
    p = np.exp(z)
    return p / p.sum()


def act(self, game_state: dict) -> str:
    ctx = context(game_state)                        # one BFS, reused six times
    phi = feature_matrix(game_state, ctx)            # (|A|, N_FEATURES)
    q = phi @ self.w

    # cached for train.py so the update never recomputes what act() already knows
    self.last_phi, self.last_q, self.last_ctx = phi, q, ctx

    p = action_probabilities(q, self.tau)

    if getattr(self, 'train', False):
        # damp exploratory self-destruction; the greedy choice is untouched
        p = p.copy()
        p[ACTIONS.index('BOMB')] *= BOMB_EXPLORE
        if game_state['round'] <= MASK_FATAL_ROUNDS:
            survivable = phi[:, NO_ESCAPE] < 0.5
            if survivable.any():
                p *= survivable
        p /= p.sum()

    idx = self.rng.choice(len(ACTIONS), p=p)
    self.logger.debug(f'q={np.round(q, 3)} -> {ACTIONS[idx]}')
    return ACTIONS[idx]