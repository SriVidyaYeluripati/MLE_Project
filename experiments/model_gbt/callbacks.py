import os
import pickle

import numpy as np

from .features import state_to_features as _state_to_features, N_FEATURES

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = "model_gbt.pt"

SEARCH_IDX = [4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 23, 24, 25, 26, 27]

USE_SEARCH = True


def state_to_features(game_state):
    f = _state_to_features(game_state)
    if f is None or USE_SEARCH:
        return f
    f = f.copy()
    f[SEARCH_IDX] = 0.0
    return f


def setup(self):
    self.n_features = N_FEATURES
    self.round_count = None
    self.epsilon_saved = None
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as file:
            saved = pickle.load(file)
        if isinstance(saved, dict):
            self.model = saved["model"]
            self.round_count = saved.get("round_count")
            self.epsilon_saved = saved.get("epsilon")
        else:
            self.model = saved
        self.logger.info(f"Loaded Model GBT from disk (search={USE_SEARCH}, round_count={self.round_count}).")
    else:
        self.model = None
        self.logger.info(f"No saved model found, starting from scratch (search={USE_SEARCH}).")


def q_values(self, features):
    if self.model is None:
        return np.zeros(len(ACTIONS))
    x = features.reshape(1, -1)
    return np.array([reg.predict(x)[0] for reg in self.model])


def act(self, game_state):
    features = state_to_features(game_state)
    if features is None:
        return "WAIT"

    eps = getattr(self, "epsilon", 0.0)
    if np.random.rand() < eps:
        return np.random.choice(ACTIONS)

    q = q_values(self, features)
    best = np.flatnonzero(q == q.max())
    return ACTIONS[np.random.choice(best)]
