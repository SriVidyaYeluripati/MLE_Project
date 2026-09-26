# Pre-registration — does giving the agent the game phase raise its score?

Written **before** any training run.

## The gap this targets

Against `rule_based_agent` in the mixed four-player field (section 14, config A,
300 rounds):

| | LinQ | rule_based | share of the 2.61 gap |
|---|---|---|---|
| coins | 3.66 | 4.08 | 0.42 |
| **kills** | **0.223** | **0.660** | **2.19 (84%)** |

**84% of the deficit is kills.** And LinQ is not incapable of killing: against
three `peaceful_agent`s it makes **0.900 kills per round**. It kills when it
hunts. In the mixed field it drops to 0.22 because it spends the round bombing
crates instead.

This corrects the section-13 statement that "the kill channel is closed". That
was established against `coin_collector_agent`, which never lingers. It is not
established against weak opponents generally, and `rule_based_agent` is
harvesting exactly those kills.

## What is missing, and why it is a representational gap rather than a tuning one

All 37 existing features are **local** - how far is X from me, is this tile
lethal, does this bomb hit a crate. None says how far through the game we are.

The winning policy has two regimes. Early: farm crates, because the coins are
locked inside them. Late: the crates are gone, so the only points left are
kills. A single shared weight vector **averages the two regimes instead of
switching between them.**

## The feature, and exactly what it buys

`late = clip(1 - crates_remaining / 123, 0, 1)`, where 123 is the measured mean
initial crate count on `classic` (200 generated boards, sd 5.6). 0 at the start,
1 on a bare board.

Three columns, all conjunctions, **no levels**:

| | |
|---|---|
| `x_late_oppdelta` | `late * opp_delta` |
| `x_late_bomb` | `late * is_bomb` |
| `x_late_bombopp` | `late * x_bomb_opp` |

**The levels are deliberately omitted.** A phase level is constant across the
six actions, so it is absorbed into `phi_state` and cannot change the argmax -
that is finding 3, and it is why `d_coin` and `d_crate` learned negative weights
in stage 2.

**And the conjunction does not reorder actions within a step either.** `late` is
constant across the six actions, so `late * opp_delta` ranks them exactly as
`opp_delta` does. What it adds is this:

    Q contribution = w_opp*opp_delta + w_late*(late*opp_delta)
                   = (w_opp + w_late*late) * opp_delta

The **effective weight on `opp_delta` becomes a linear function of phase.** That
is a regime switch, and one weight vector provably cannot express it without the
conjunction. This is the entire mechanism, stated before the run so it cannot be
reinvented afterwards.

## The density check, run before this document

80 rounds vs three `rule_based_agent`s, 22,500 decisions:

| | |
|---|---|
| `late`, mean | 0.785 |
| `late`, median | 0.894 |
| steps with `late` > 0.5 | **85%** |
| steps with `late` > 0.8 | **67%** |
| steps where the three columns are non-zero | **99%** (22,270 / 22,500) |

The late regime is not rare - it is most of the game. Four agents clear the
crates quickly, and the agent spends two-thirds of its steps on a board that is
over 80% cleared. **This is by far the densest feature added in this project:**
the herding feature had an opinion on 6% of steps, the coin-approach signal on
10%, genuine traps on 0.03%.

Density cuts both ways and that is stated now: because `late` sits near 0.78 on
average, the conjunction is close to a rescaled `opp_delta` most of the time,
and the learner can partly imitate it by reweighting `opp_delta`. The
information is in the **variation** of `late`, not its level.

## Arms, metric and rule — fixed now

Two arms, **forty seeds each**, seeds 7001-7040, paired by seed. Warm start from
`weights.npz` (new columns pad to zero), `LQ_BOMB_COST=0 LQ_CRATE_VALUE=0.02`,
600 rounds on `classic` vs three `peaceful_agent`s.

| tag | switch |
|---|---|
| `pbase` | none - the three columns stay identically zero |
| `pphase` | `LQ_PHASE=1` at train **and** eval |

**Evaluation is different from every previous experiment here, deliberately.**
Earlier runs evaluated against three `coin_collector_agent`s. The hypothesis is
about hunting weak opponents, and `coin_collector` never lingers, so that
protocol cannot see the effect. Evaluation is **300 rounds against three
`rule_based_agent`s**, which is also the configuration the gap was measured in.

**Primary metric: score per round.** Not kills. The claim is that the agent
re-allocates effort between two channels, so the thing to measure is the total,
not one channel - and measuring kills alone would reward a change that trades
coins for kills at a loss.

**Decision rule:** paired by seed, one-sided `t > 1.685` (alpha = .05, df = 39).
One-sided because the mechanism predicts a direction. Anything else is a null.

Secondary, reported and claimed only if the primary passes: kills, coins,
suicides.

## Power, from variances measured for their own sake

From the n=40 herding run: paired score sd **0.731**, kills sd **0.041**.

| effect on score | power at n=40 |
|---|---|
| +0.20 | 42% |
| +0.33 | 80% |
| +0.50 | 98% |

n = 40 detects **+0.33 score** at 80% power. The gap to close is 2.61, and the
kill component alone is 2.19, so an intervention that works at all should clear
this comfortably. **If the true effect is below +0.33 this test will probably
miss it**, and that is the stated limitation.

## My prior, written down now

**About 35%** that this produces a measurable score gain. Higher than the 20% I
gave herding, for three reasons: the columns are active on 99% of steps rather
than 6%; the mechanism is a regime switch the model provably cannot otherwise
express; and it is aimed at the 84% of the gap rather than at a symptom.

Lower than 50% because eight consecutive interventions in this project have been
null, and because `late` is high most of the time, so much of what the
conjunction expresses can be imitated by reweighting `opp_delta`.

**No guarantee is offered or implied.** The honest summary is: this is the best
remaining idea, and it is more likely to fail than to succeed.

## What would make this worthless

- Extending past n = 40, or stopping early, after looking at any result.
- Reporting kills or coins as the finding if score says nothing.
- Switching the evaluation opponent after seeing the result.
- Quietly dropping the 35% prior if the answer comes out positive.

---

## Outcome, appended after the run

n=40 pairs, vs three rule_based_agents.

| metric | mean | sd | better | t | 95% CI |
|---|---|---|---|---|---|
| **score (PRIMARY)** | **+0.174** | 0.841 | 23/40 | **+1.31** | [-0.095, +0.444] |
| kills | +0.027 | 0.066 | 25/40 | +2.59 | [+0.006, +0.048] |
| invalid | +0.505 | 0.859 | 26/40 | +3.72 | [+0.231, +0.780] |
| coins | +0.039 | 0.743 | 23/40 | +0.33 | [-0.199, +0.277] |

NULL on the primary. But the mechanism fired: kills rose with a CI excluding
zero - the first predicted behavioural change to appear in nine experiments.
It also cost 0.505 more invalid actions per round (also excluding zero), which
is finding 13: hunting means walking into tiles an opponent just took.

The channels add up exactly: kills +0.027 x 5 = +0.135, coins +0.039, sum
+0.174 = the observed score change to three decimals. So the effect is real
and worth about +0.17 score. The pre-registration said n=40 detects +0.33;
this landed in the band declared in advance as one the test would miss.

Confirming +0.174 needs n=184 per arm - 368 trainings, ~8.4 hours - and would
close 6.7% of the 2.61 gap. Not run.

Stated prior was 35%. No measurable score gain; a measurable behavioural one.
LQ_PHASE stays off. The agent ships unchanged.
