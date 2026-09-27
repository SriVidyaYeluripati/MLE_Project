Model GBT — scripts and weights

Fitted Q-iteration with gradient-boosted trees, sharing model_linearQ's 28-feature
contract for a controlled comparison.

Note on the weights/ folder: unlike LinearQ, most of GBT's early experiment
checkpoints were overwritten during live debugging rather than kept one-per-run.
Only the four surviving checkpoints are here. Full before/after numbers for every
attempt, including the ones with no surviving weight file, are in the report log.

weights/model_gbt_coinheaven.pt — coin-heaven only, 1000/1000 coins, 0 mistakes
weights/model_gbt_classic.pt — classic mode, early fix, before route-count scaling
weights/model_gbt_classic_v2.pt — classic mode, after route-count scaling
weights/final_classic.pt — current best, after exploration-persistence fix and
conservative 2-route bombing rule
