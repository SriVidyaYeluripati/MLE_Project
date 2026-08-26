"""
Model A - callbacks.py  (the part that PLAYS)

Model A is Q-learning with one regressor per action (fitted Q-iteration).
It uses lecture techniques - regression forests / ridge regression - rather
than a neural network, which satisfies the project's "lecture techniques"
requirement and acts as our always-submittable fallback.
"""
import os
import pickle
import numpy as np

from .features import state_to_features, N_FEATURES

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = "model_a.pt"


def setup(self):
    """Called once when the agent loads. Load the six regressors if present."""
    self.n_features = N_FEATURES
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
        self.logger.info("Loaded Model A from disk.")
    else:
        self.model = None          # untrained -> Q = 0 for every action
        self.logger.info("No saved model found, starting from scratch.")


def q_values(self, features):
    """Q(s, a) for all six actions. An untrained model returns zeros."""
    if self.model is None:
        return np.zeros(len(ACTIONS))
    x = features.reshape(1, -1)
    return np.array([reg.predict(x)[0] for reg in self.model])


def act(self, game_state):
    """Called every step. Must return within 0.5 s when not training."""
    features = state_to_features(game_state)
    if features is None:
        return "WAIT"

    # epsilon is set by train.py during training; 0.0 when just playing
    eps = getattr(self, "epsilon", 0.0)
    if np.random.rand() < eps:
        return np.random.choice(ACTIONS)

    q = q_values(self, features)
    # random tie-break so an untrained agent doesn't always pick UP
    best = np.flatnonzero(q == q.max())
    return ACTIONS[np.random.choice(best)]
