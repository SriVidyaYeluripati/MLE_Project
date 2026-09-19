# Pre-registration — the bombing experiment

Written **before** the runs. The point is that the arms, the primary metric and
the decision rule are fixed in advance, because three results in this project
were reported and then withdrawn after being chosen post hoc from noisy data
(build log findings 15, 16, 17).

## The diagnosis this tests

`CRATE_DESTROYED` pays `CRATE_VALUE = 0.02` per crate, but arrives four steps
after the bomb that caused it (`BOMB_TIMER = 4`). The accumulating trace decays
by `gamma * lambda = 0.95 * 0.80 = 0.76` per step, so the bombing action retains
`0.76^4 = 0.334` of it. The shipped weights are exactly what that predicts:

| feature | predicted | measured |
|---|---|---|
| `crates_hit_1` | 1 x 0.02 x 0.334 = 0.0067 | +0.0073 |
| `crates_hit_3p` | 4 x 0.02 x 0.334 = 0.0267 | +0.0274 |
| `is_bomb` (paid immediately, for the act) | — | **+0.0348** |

So the agent is paid ~5x more for pressing BOMB than for hitting one crate, and
the learner is fitting that signal correctly. Measured consequence: 46.6 bombs
per round at 0.43 crates per bomb, against `coin_collector`'s 11.9 at 2.89.

Two further facts the arms are built on:

- `bomb_value[pos]` — the exact number of crates a bomb here destroys — is
  already computed at decision time by `context()`. The delay is avoidable.
- `bomb_value >= 1` selects **99.9%** of free tiles, so `d_crate` is constant
  and `crate_delta` trained to −0.0008. There is no "walk to a good bombing
  spot" gradient at all.

## Arms — 5, four seeds each, 20 trainings

Every arm: warm start from `weights.npz`, 600 rounds on `classic` against three
`peaceful_agent`s, `LQ_BOMB_COST=0 LQ_CRATE_VALUE=0.02`, seeds 2001–2004.

| tag | switch | hypothesis |
|---|---|---|
| `base` | none | control — reproduces §9's recipe, whose spread is known to be 1.68 margin |
| `drop` | `LQ_DROP_REWARD=1` | pay for crates at drop time; removes the 0.334 discount |
| `tgt` | `LQ_BOMB_TARGET=1` | narrow the crate target set to the best tiles; creates a gradient |
| `both` | both of the above | do they compose, or is one sufficient? |
| `filt` | `LQ_BOMB_FILTER=1` at train **and** eval | learn inside the constraint rather than bolting it on |

## Primary metric — fixed now

**`crates per bomb`, and `coins`.** Not margin.

Margin is score minus the best opponent; score is coins plus five times kills;
kills are rare, so margin inherits a rare event's variance. Measured in this
project: coins replicate to ±0.05, margin to ±0.10, and *training* spread on
margin is 1.68. Crates per bomb is an average over ~40 events per round and is
the quantity the intervention acts on directly.

Margin is still reported, as a secondary, with its spread stated.

## Decision rule — fixed now

An arm is called an improvement only if **the four-seed mean beats `base`'s
four-seed mean by more than the larger of the two arms' standard deviations**,
on crates per bomb *and* without a fall in coins.

If nothing clears that, the answer is "no effect", and it gets written up as
one. A negative result here is a perfectly good outcome: it would say the
bombing waste is not fixable by reward design, and that the play-time filter
(§10, already measured) is the right answer.

## Evaluation

300 rounds against three `coin_collector_agent`s, learning off, one evaluation
per trained weight vector. Repeating an evaluation is near-worthless here —
§9 measured evaluation noise at ~0.1 against training noise of 1.68 — so the
budget goes into seeds, not replicates.

## What would make this experiment worthless

Stated in advance so it can be checked afterwards:

- Reporting the best seed of an arm instead of its mean.
- Switching to margin if crates-per-bomb says nothing.
- Adding an arm after seeing the results and reporting it alongside these.
- Declaring an effect that does not clear the decision rule above.
