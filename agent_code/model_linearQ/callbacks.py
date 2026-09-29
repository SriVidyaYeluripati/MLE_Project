
import os

import numpy as np

from .features import (ACTIONS, N_FEATURES, FEATURE_NAMES, BIAS, IS_INVALID,
                       feature_matrix, context)

MODEL_FILE = 'weights.npz'

OPTIMISTIC_INIT = 0.5
TAU_TRAIN = 0.25
TAU_PLAY = 0.0
TIE_EPS = 1e-9

BOMB_EXPLORE = 0.25

def model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)

def setup(self):
    #Called once before the first round.
    path = model_path()
    self.rng = np.random.default_rng()

    if os.path.isfile(path):
        data = np.load(path, allow_pickle=True)
        w = data['w'].astype(float)

        # weights are indexed by POSITION, so refuse a file whose columns differ
        saved = [str(x) for x in data['feature_names']]
        if saved != FEATURE_NAMES[:len(saved)]:
            raise ValueError(f'{MODEL_FILE} was trained on different features; '
                             f'retrain or restore the matching file')

        if len(w) < N_FEATURES:
            self.logger.warning(f'weights have {len(w)} of {N_FEATURES} features'
                                f' - padding the new ones with zeros (warm start)')
            w = np.concatenate([w, np.zeros(N_FEATURES - len(w))])
        elif len(w) > N_FEATURES:
            raise ValueError(f'{MODEL_FILE} has {len(w)} features, '
                             f'code has {N_FEATURES}')
        self.w = w
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
#legal actions are marked with 1 in the IS_INVALID column of the feature matrix.
    return phi[:, IS_INVALID] < 0.5


def action_probabilities(q, tau, legal=None):
#we are using a softmax function to convert Q-values into action probabilities. 

    q = np.asarray(q, dtype=float)
    if legal is None:
        legal = np.ones(len(q), dtype=bool)
    else:
        legal = np.asarray(legal, dtype=bool)
        if not legal.any():
            legal = np.ones(len(q), dtype=bool)

    q_legal = q[legal]
    p = np.zeros(len(q))
    #The temperature tau controls the exploration-exploitation trade-off.
    # so if tau is 0, we choose the action with the highest Q-value among legal actions.
    if tau <= 0:
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
# Obviously , here we are using the learned Q-values to select an action based on the current game state. 
    ctx = context(game_state)
    phi = feature_matrix(game_state, ctx)
    q = phi @ self.w

    self.last_phi, self.last_q, self.last_ctx = phi, q, ctx

    p = action_probabilities(q, self.tau, legal_mask(phi))

    if getattr(self, 'train', False):
        p = p.copy()
        p[ACTIONS.index('BOMB')] *= BOMB_EXPLORE
        p /= p.sum()

    idx = self.rng.choice(len(ACTIONS), p=p)
    self.logger.debug(f'q={np.round(q, 3)} -> {ACTIONS[idx]}')
    return ACTIONS[idx]