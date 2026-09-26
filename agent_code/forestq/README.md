# ForestQ — fitted Q-iteration with regression forests

Owner: Sri Vidya Yeluripati.

Six regressors, one per action, fitted on collected transitions; the targets are
rebuilt as r + gamma * max Q(s', a') after each batch and all six are refitted.
Random forests of 40 trees, minimum leaf size 5, gamma 0.9, epsilon decaying
from 1.0 to 0.05 over 400 rounds, and 8x data augmentation from the board
symmetries.

## The shared feature contract

`features.py` holds the 28-feature contract used by this agent and by
`agent_code/model_gbt`. Anyone who changes `FEATURE_NAMES` must tell the team,
because it changes the input of every model that uses it.

## Files

- `features.py`   the 28-feature contract
- `callbacks.py`  play: the six regressors score the six actions, best one wins
- `train.py`      fitted Q-iteration with symmetry augmentation
- `forestq.pt`    trained model, six forests, pickled with scikit-learn 1.9.0

## Install

    pip install numpy scikit-learn
