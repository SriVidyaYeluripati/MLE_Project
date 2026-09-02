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

# Exploration is NOT uniform over the six actions.  A random BOMB can end the
# episode, and an episode that ends early stops producing data - so uniform
# exploration systematically collects less experience per unit of wall clock,
# and biases what it does collect toward the consequences of careless bombs.
# That reasoning fits an agent that bombs too much.  Model A bombs too LITTLE
# (0.22 per round), so it needs bomb experience, not less of it: equal share.
EXPLORE_P = np.array([1., 1., 1., 1., 1., 1.0])
EXPLORE_P = EXPLORE_P / EXPLORE_P.sum()


def setup(self):
    """Called once when the agent loads. Load the six regressors if present."""
    self.n_features = N_FEATURES
    if os.path.isfile(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            self.model = pickle.load(f)
        _make_prediction_fast(self.model)
        self.logger.info("Loaded Model A from disk.")
    else:
        self.model = None          # untrained -> Q = 0 for every action
        self.logger.info("No saved model found, starting from scratch.")


def _make_prediction_fast(model):
    """
    n_jobs is pickled with the forest, so a model trained with n_jobs=-1 also
    PREDICTS with n_jobs=-1.  Every act() then spins up a joblib pool six times
    to score a single row, and the pool costs far more than the trees do:

        n_jobs=-1   89.95 ms per act()   ->  36.0 s per 400-step round
        n_jobs= 1   19.95 ms per act()   ->   8.0 s per 400-step round

    Predictions are bit-identical either way - n_jobs only controls parallelism -
    so this is free.  It matters because the task sheet allows 0.5 s per step on
    a machine we do not control, and 90 ms leaves far less headroom than 20 ms.
    """
    for reg in model:
        if hasattr(reg, 'n_jobs'):
            reg.n_jobs = 1


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
        return np.random.choice(ACTIONS, p=EXPLORE_P)

    q = q_values(self, features)
    # random tie-break so an untrained agent doesn't always pick UP
    best = np.flatnonzero(q == q.max())
    return ACTIONS[np.random.choice(best)]
