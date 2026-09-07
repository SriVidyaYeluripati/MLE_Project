"""
Always loaded, including in the tournament.   Keep this file fast and free of
any training-only imports: train.py does not exist at tournament time.
"""

import os

import numpy as np

try:
    from .features import (ACTIONS, N_FEATURES, BIAS, NO_ESCAPE, IS_INVALID,
                           feature_matrix, context)
except ImportError:
    from features import (ACTIONS, N_FEATURES, BIAS, NO_ESCAPE, IS_INVALID,
                          feature_matrix, context)

# LQ_WEIGHTS lets an ablation run write its own file instead of clobbering the
# shipped weights.  Relative to THIS file - never absolute.
MODEL_FILE = os.environ.get('LQ_WEIGHTS', 'weights.npz')

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

# An illegal action is not a mistake the agent has to LEARN about: it is a move
# into a wall, or BOMB with no bomb.  The framework leaves the agent where it
# stood, so its afterstate is identical to WAIT's - the two differ only by the
# columns is_wait and is_invalid.  Whenever the learner ends up with
#     w[is_invalid] > w[is_wait]
# an illegal move becomes a strictly cheaper way to wait, and a *greedy* policy
# will then take one every time waiting is right.  That is why the invalid
# count barely moved when we went greedy: those actions were never exploration.
# Masking removes the whole failure mode instead of hoping the weights order
# themselves correctly (Huang & Ontanon 2022, invalid action masking).
MASK_INVALID = os.environ.get('LQ_MASK_INVALID', '1') != '0'    # ablate this.


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


def legal_mask(phi):
    """Rows of a feature matrix whose action the framework will actually run."""
    return phi[:, IS_INVALID] < 0.5


def action_probabilities(q, tau, legal=None):
    """
    Scale-aware softmax over the LEGAL actions.  A fixed temperature is NOT
    scale free: early in training all Q-values sit near zero (near-uniform) and
    late they are far apart (near-greedy), so the exploration rate would anneal
    itself in a way we never chose.  Dividing by the spread fixes that.

    `legal` is a boolean mask; illegal actions get probability 0 and are left
    out of both the max and the spread, so masking cannot silently rescale the
    temperature.  WAIT is always legal, so the mask is never empty.
    """
    q = np.asarray(q, dtype=float)
    if legal is None or not MASK_INVALID:
        legal = np.ones(len(q), dtype=bool)
    else:
        legal = np.asarray(legal, dtype=bool)
        if not legal.any():
            legal = np.ones(len(q), dtype=bool)

    q_legal = q[legal]
    p = np.zeros(len(q))

    if tau <= 0:                                     # greedy, ties shared
        p[legal & (q >= q_legal.max() - TIE_EPS)] = 1.0
        return p / p.sum()

    spread = q_legal.std()
    if spread < 1e-8:
        p[legal] = 1.0
        return p / p.sum()
    z = (q - q_legal.max()) / (tau * spread)
    p[legal] = np.exp(z[legal])
    return p / p.sum()


def act(self, game_state: dict) -> str:
    ctx = context(game_state)                        # one BFS, reused six times
    phi = feature_matrix(game_state, ctx)            # (|A|, N_FEATURES)
    q = phi @ self.w

    # cached for train.py so the update never recomputes what act() already knows
    self.last_phi, self.last_q, self.last_ctx = phi, q, ctx

    p = action_probabilities(q, self.tau, legal_mask(phi))

    if getattr(self, 'train', False):
        # damp exploratory self-destruction; the greedy choice is untouched
        p = p.copy()
        p[ACTIONS.index('BOMB')] *= BOMB_EXPLORE
        if game_state['round'] <= MASK_FATAL_ROUNDS:
            survivable = phi[:, NO_ESCAPE] < 0.5
            if (p * survivable).sum() > 0:
                p *= survivable
        p /= p.sum()

    idx = self.rng.choice(len(ACTIONS), p=p)
    self.logger.debug(f'q={np.round(q, 3)} -> {ACTIONS[idx]}')
    return ACTIONS[idx]