# Pre-registration — can a confinement feature make the agent create traps?

Written **before** any training run. The redundancy check that justified the
feature ran first and is in `experiments/herd_probe.py`; its numbers are quoted
below and were known before this document was written.

## What §13 established, and why this is the remaining channel

The kill channel is capped by **opportunity, not policy**. Genuine trap
opportunities arise **0.115 times per round** and the agent already takes
**21 of 23** of them. So raising kills cannot come from converting better; it
must come from *creating* trap geometry — herding.

No existing feature can express that. `OPP_IN_BLAST` and `OPP_TRAPPED` are
assigned inside `if is_bomb and ctx['bombs_left']` (features.py), so they are
**zero on every movement row**. `OPP_DELTA` says only "I got closer". Nothing
says "standing here takes room away from them".

## The feature

`opp_confine`, feature 36, on the movement rows, zero unless `LQ_HERD=1`:

    1 - (tiles the nearest opponent can reach, capped at HERD_CAP,
         given that I am standing on this destination) / HERD_CAP

computed only when an opponent is within `HERD_NEAR = 8` steps. `HERD_CAP = 60`.

Verified on constructed boards before any training:

| board | `opp_confine` across the six actions |
|---|---|
| opponent in the open | 0, 0, 0, 0, 0, 0 |
| opponent in a pocket | 0, 0.950, **0.983**, 0, 0.967, 0.967 |

Zero when nobody can be confined, and a spread of 0.98 when they can — so the
column carries a real gradient rather than a constant, which is the failure that
killed `crate_delta` (finding: `bomb_value >= 1` selected 99.9% of tiles, so the
feature trained to -0.0008).

## The redundancy check, run before this document

45,600 steps of real play plus a horizon sweep (`experiments/herd_probe.py`):

| CAP | has any gradient | disagrees with `opp_delta` | genuinely new |
|---|---|---|---|
| 25 | 4.5% of steps | 40% | 1.8% |
| **60** | **6.2%** | 36% | **2.2%** |
| 120 | 3.6% | 38% | 1.4% |

**Not redundant** — the rejected artificial potential field (finding 24) agreed
with the existing `d_coin` 98.6-100% of the time. This disagrees with
`opp_delta` on about 37% of the steps where it has an opinion.

**But sparse**, and that is the honest prior: it has an opinion on 6% of steps,
and it agrees with `exits` 86% of the time, which is the tell — cornering in
Bomberman is mostly done by walls, and walls are already in the feature vector.
For scale: the coin-approach signal is available on 10% of steps and is
learnable; genuine traps arise on ~0.03% and needed a curriculum (finding 7).

**My stated prior: about 20% that this shows anything.** Written down now.

## Arms, metric and rule — fixed now

Two arms, **forty seeds each**, seeds 6001-6040, paired by seed. Warm start
from `weights.npz` (the new column pads to zero), `LQ_BOMB_COST=0
LQ_CRATE_VALUE=0.02`, 600 rounds on `classic` vs three `peaceful_agent`s;
evaluated 300 rounds vs three `coin_collector_agent`s, learning off.

| tag | switch |
|---|---|
| `hbase` | none — `opp_confine` stays identically zero |
| `hherd` | `LQ_HERD=1` at train **and** eval |

**Primary metric: kills per round.** One-sided, because the mechanism predicts
a direction. This is forced by the arithmetic, not chosen for convenience —
see the power section.

**Decision rule:** paired by seed, `t > 1.685` (one-sided, alpha = .05,
df = 39). Anything else is a null and gets written up as one.

Secondary, reported with the primary and claimed only if the primary passes:
score, coins, suicides.

## Power, and why n = 40 and why kills

From variances measured for their own sake in the n=10 confirmatory run —
kills 0.087, coins 0.748, score 1.004 (finding 26: never take these from the
run that motivated the experiment):

| to detect | metric | n per arm needed |
|---|---|---|
| **+0.04 kills** — the entire deficit vs the best `coin_collector` | kills | **31** |
| +0.20 score — that same deficit expressed in score | score | 158 |

n = 40 per arm detects **+0.039 kills** at 80% power, which is almost exactly
the gap worth closing. **A score-primary version of this experiment is not
runnable** — 316 trainings — and that is why kills is primary. Stated in
advance so the choice cannot look like a post-hoc convenience.

n is locked at 40 now and will not be extended after looking.

## Mechanism check, registered as a secondary

If the feature works by making the agent seek trap geometry, then trap
*opportunities* should rise, not just conversions. The probe measures this
directly: genuine traps per round, currently **0.115**, and the take-up rate,
currently **21 of 23**. Registered prediction:

> `hherd` shows more genuine trap opportunities per round than `hbase`,
> at a roughly unchanged take-up rate.

If kills rise with opportunities flat, the effect is real but the stated
mechanism is wrong, and it must be reported that way.

## What would make this worthless

- Extending past n = 40, or stopping early, after looking at any result.
- Reporting score or coins as the finding if kills says nothing.
- Reporting the feature as validated because the *probe* showed it is not
  redundant. Non-redundant is a precondition for the experiment, not a result.
- Quietly dropping the stated 20% prior if the answer comes out positive.

---

## Outcome, appended after the run

Run 2026-09-09, all 40 seed pairs, 80 trainings and 80 evaluations complete.

| paired, hherd minus hbase, n=40 | mean | sd | better in | t | 95% CI |
|---|---|---|---|---|---|
| **kills (PRIMARY)** | **-0.003** | 0.041 | 15/40 | **-0.39** | [-0.016, +0.010] |
| score | +0.167 | 0.731 | 22/40 | +1.44 | [-0.067, +0.400] |
| coins | +0.180 | 0.618 | 23/40 | +1.84 | [-0.018, +0.377] |
| suicides | -0.002 | 0.063 | 19/40 | -0.23 | [-0.023, +0.018] |
| invalid | -0.050 | 0.203 | 16/40 | -1.56 | [-0.115, +0.015] |
| margin | +0.228 | 0.933 | 21/40 | +1.55 | [-0.070, +0.527] |

**NULL on the pre-registered primary**, and this time a *tight* one.

At n=40 the test detects **+0.018 kills** at 80% power against a target
deficit of **+0.040** - it could have seen an effect **2.2x smaller than the
one worth having** and found none. CI width on kills is 0.026, against 1.070
for the section-14 coins test. That one could not exclude a real effect;
this one can. "We found nothing" and "there is nothing to find" are
different claims, and only a powered design earns the second.

**Secondaries are not claimed.** coins +0.180 (t=1.84) and score +0.167
(t=1.44) arrived in an experiment whose primary failed, and this document
fixed in advance that secondaries are claimed only if the primary passes.
Third positive-looking coins secondary correctly refused.

**The stated 20% prior held**, and the redundancy probe predicted it: the
feature has an opinion on ~6% of steps and agrees with `exits` 86% of the
time, because cornering is done by walls and walls were already represented.

**What this closes.** Section 13 showed the kill channel is capped by
opportunity - 0.115 genuine traps per round, of which the agent takes 21 of
23. Herding was the last way to raise that number. It does not. Channel
closed; the agent ships unchanged with LQ_HERD off.
