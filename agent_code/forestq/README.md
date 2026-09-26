# ForestQ — fitted Q-iteration with regression forests

Owner: Sri Vidya Yeluripati.

ForestQ is six regressors, one per action, fitted on collected transitions.
After each batch we rebuild the targets as r + gamma * max Q(s', a') and
refit all six. Each regressor is a random forest of 40 trees, minimum leaf
size 5. Training uses gamma = 0.9, epsilon decaying from 1.0 to 0.05 over 400
rounds, and 8x data augmentation from the board's rotational and mirror
symmetries.

## The shared feature contract

`features.py` holds the 28-feature contract that this agent and
`agent_code/model_gbt` both use. If you change `FEATURE_NAMES`, tell the
team first — it changes the input of every model built on this contract, not
just this one.

## Files

- `features.py`   the 28-feature contract
- `callbacks.py`  play: the six regressors score the six actions, best one wins
- `train.py`      fitted Q-iteration with symmetry augmentation
- `forestq.pt`    trained model, six forests, pickled with scikit-learn 1.9.0

## Install

    pip install numpy scikit-learn