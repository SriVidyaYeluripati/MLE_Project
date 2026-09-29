# Pre-registration — does a shorter credit horizon help a weak-feature linear agent?

Written **before** any run. Committed before the first training launches.

## The claim being tested

nickstr15 ("Maverick", Heidelberg WS20/21, 23 features) report in their project
report that **gamma = 0.85 produced endless movement loops and gamma = 0.6 fixed
them**, and that lowering gamma was necessary — a terminal-state fix alone did
not stop the loops.

Their mechanism, which is the reason this is worth a run rather than a shrug:

> with features too weak to linearly express the true remaining return, the
> best linear fit to a long-horizon target is degenerate. Shrink the horizon to
> what the features can actually see.

Model LinQ trains at **gamma = 0.95**, i.e. an effective horizon of
`1/(1-gamma) = 20` steps. Its features see much less than 20 steps: `d_coin`
and `d_safety` are `0.9^steps`, which is under 0.12 by step 20, and the
survivability recursion looks 4 steps ahead. So the premise applies here.

## The honest prior, stated now

**Model LinQ does not have the symptom their fix addresses.** Their agent
looped; ours does not — 0.4 invalid actions per round (findings 12-13 fixed the
ordering bug that caused wasted steps) and 304 steps per round against a 400
limit. A published fix is a hypothesis about *that* agent's failure mode.
Section 14 already recorded one such transfer failing: chbridges' deployment-
exploration fix does nothing here, because our invalid-action mask had already
removed the oscillation it targets.

So this is a **mechanism check, not an expected win.** Recorded now so a null
is not reinterpreted afterwards as "we knew that".

## Arms, metric and rule — fixed now

Three arms, **twelve seeds each**, seeds 5001-5012, paired by seed. Warm start
from `weights.npz`, `LQ_BOMB_COST=0 LQ_CRATE_VALUE=0.02`, 600 rounds on
`classic` vs three `peaceful_agent`s; evaluated 300 rounds vs three
`coin_collector_agent`s, learning off, one evaluation per weight vector.

| tag | switch |
|---|---|
| `g95` | none (control, gamma = 0.95) |
| `g80` | `LQ_GAMMA=0.80` |
| `g60` | `LQ_GAMMA=0.60` |

**Primary comparison: `g60` vs `g95`, on score per round.** One comparison, not
three — `g80` is an intermediate point, reported for shape and *not* tested.
Score rather than coins because gamma changes the whole value function, not one
channel.

**Decision rule:** paired by seed, two-sided `t > 2.201` (df = 11). Two-sided
because, unlike the herding test, there is no directional prediction: their
result predicts improvement, the prior above predicts nothing or harm.

Secondary, reported alongside: coins, kills, suicides, steps, invalid actions.
`invalid` and `steps` matter because they are where a loop would show.

## Power, from a variance measured for its own sake

Using the paired score sd of **1.004** from the n=10 confirmatory run
(`PREREG_trapped_confirm.md`) — a control condition measured separately, which
is finding 26 applied rather than repeated:

| effect on score | power at n=12 |
|---|---|
| +0.50 | 30% |
| +0.80 | 62% |
| +1.00 | 80% |
| +1.50 | 98% |

**This test can only find a large effect (>= ~1.0 score),** which is the right
size for the claim: Maverick describe the difference between an agent that
loops and one that does not. A null bounds the effect at roughly +-1.0 and
should be reported as "no large effect", never as "no effect".

## What would make this worthless

- Testing all three pairwise comparisons and reporting the significant one.
- Switching to coins, margin or kills if score says nothing.
- Extending past 12 seeds after looking.
- Reporting `g80` as a result rather than as an intermediate point.
