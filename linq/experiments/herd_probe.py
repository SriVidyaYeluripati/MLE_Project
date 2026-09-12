"""
Redundancy check for a proposed HERDING feature, run BEFORE any training.

Finding 24: ask whether a new feature says anything different from the ones
already there, offline, because after training a redundant feature produces
exactly the same evidence as a useful one.

The candidate, `opp_confine`:
    after I step onto `dest`, how confined is the nearest opponent?
    = 1 - (tiles that opponent can reach, capped at CAP, with my body at dest)
          / CAP
It is 0 when they are in the open and approaches 1 as they are cornered.
This is the "create a trap" signal that OPP_TRAPPED cannot be, because
OPP_TRAPPED is assigned only on the BOMB row (features.py:481) and so is zero
on every movement row.

Three questions, in the order that can kill the idea cheapest:
  Q1  does it VARY across the legal moves?  (d_crate did not: bomb_value >= 1
      selected 99.9% of tiles, so `crate_delta` trained to -0.0008)
  Q2  when it varies, does its best move differ from `opp_delta`'s best move?
      opp_delta already means "get closer to the nearest opponent", and
      cornering usually requires closing in, so this is the real test.
  Q3  does it differ from `exits`/`dead_end`, which describe MY destination?
"""
import json, os, numpy as np
from collections import deque

import os as _os
CAP = int(_os.environ.get('LQ_HERD_CAP', '25'))   # cells; beyond this, "in the open"
NEAR = 8            # only ask the question when an opponent is within 8 steps

def region_size(start, free, blocked_extra, cap=CAP):
    """How many free tiles can `start` reach, counting at most `cap`."""
    if not free[start]:
        return 0
    seen = {start}
    dq = deque([start])
    while dq and len(seen) < cap:
        x, y = dq.popleft()
        for n in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if n in seen or n == blocked_extra:
                continue
            if free[n]:
                seen.add(n)
                dq.append(n)
    return len(seen)

def confine(dest, field, opp, cap=CAP):
    free = field == 0
    return 1.0 - region_size(opp, free, dest, cap) / cap


# --------------------------------------------------------------------------- #
# RESULT, measured before any training was launched.
#
# 45,600 steps vs 3x coin_collector_agent, plus a horizon sweep.
#
#   CAP   varies/usable   varies/all   agrees with opp_delta   NEW signal
#    25       22.2%          4.48%            60.4%              1.77%
#    60       35.7%          6.20%            63.8%              2.24%   <- best
#   120       21.3%          3.61%            62.5%              1.35%
#
# VERDICT: not redundant, but sparse.
#
# Not redundant: the artificial potential field (finding 24) picked the same
# direction as the existing d_coin 98.6-100% of the time and was rejected on
# that.  This picks a different direction from opp_delta ~37% of the times it
# has an opinion, so it is a genuinely new signal.
#
# Sparse: it has any gradient at all on only ~6% of steps, and carries new
# direction on ~2.2%.  For scale: the coin-approach signal is available on 10%
# of steps and is learnable; genuine trap opportunities arise on ~0.03% and
# needed a curriculum (finding 7).  This sits between.
#
# Why it is flat so often: one body rarely changes how much room an opponent
# has.  Cornering somebody in Bomberman is mostly done by the WALLS, which are
# already described by `exits` and `dead_end` - and indeed the best move here
# agrees with `exits` 86% of the time at CAP=25.
#
# CONSEQUENCE FOR THE EXPERIMENT DESIGN: the effect this could produce is a
# kill gain, not a score gain.  Powered from the sd measured in the n=10
# confirmatory run (kills 0.087, score 1.004):
#
#   detect +0.04 kills (the whole deficit vs coin_collector)   n = 31 per arm
#   detect +0.20 score (that same deficit expressed in score)  n = 158 per arm
#
# So kills MUST be the primary metric.  A score-primary version of this
# experiment is not runnable: it would need 316 trainings.
