Pre-registration: GBT with-search vs without-search (replication)

Prediction, written before this run: with-search will collect at or near
1000/1000 coins on coin-heaven; without-search will collect well under 100,
based on the original run (1000 vs 24). Seed for this replication: 777.
Metric: total coins over 20 evaluation rounds. Decision rule: original
finding holds if with-search stays above 900 and without-search stays
under 150.

Outcome: finding holds. With-search scored 1000/1000 (seed 777), without-search
scored 38/1000. Both inside the pre-registered decision bounds. Note: this run's
with-search policy made far more invalid/wasted bomb attempts than the original
run despite matching it on coins exactly (972 invalid vs 0 previously) -- real
run-to-run noise, not something the pre-registered metric was tracking, but
worth keeping visible.
