# Pre-registration — B2, a control condition for task 2

Registered **before** any run. Model LinQ, `experiments-v1` worktree.

## This is a measurement, not a test

There is no hypothesis and no decision rule, because nothing is being compared.
**The purpose is to measure how much two identical agents differ on task 2**, so
that later experiments have a threshold instead of a guess.

E29 did exactly this for the tournament configuration — six identical trainings,
spread 1.68 margin — and that number went on to retract three claims (E30a, E30b,
E30c). **Task 2 has never had the equivalent.** Every task-2 result in this
project (E63, E64, E65, E66) is currently read against a spread *inferred* from
E65's baseline arm, which was designed as an experimental arm and not as a
control.

## What is measured

**Coins per round on task 2, across six identical trainings.**

Reported: mean, sd, min, max, and the full six values. Nothing else is claimed.

## Design

- **n:** 6 trainings, fixed before any data exists.
- **Identical in every respect.** No ablation, no switches, no reward changes.
  The six differ only through the framework's unseeded randomness — the same
  source of variation E29 measured.
- **No training seed is passed.** Fixing the world seed would suppress part of
  the very variance being measured.
- **Training protocol — identical to E66's control arm**, so the spread applies
  directly to E66 and to A3: 400 rounds solo on `classic`, then 300 rounds on
  `classic` vs 3× `peaceful_agent`, from scratch, each run writing its own
  `LQ_WEIGHTS` file.
- **Evaluation protocol — identical to E63/E66:** solo `classic`, 100 rounds,
  learning off, `--seed 1` for all six so every run faces the same boards.
  Evaluation boards are held fixed deliberately: the quantity of interest is
  **training** variance, and E26/E29 established that it is roughly 10× the
  evaluation variance.

## What this will be used for

1. **To set A3's decision threshold.** A3 asks whether `conj` and `mask` cost
   task 2. E66 measured +3.57 and +2.51 at n = 1 and both were refused for want
   of a spread. This supplies it — and it must be measured **before** A3 runs,
   because powering a test of E66 using E66's own single-seed gaps is E47's
   error.
2. **To re-read E63–E66 against a threshold** rather than a caveat.

## What would make this experiment worthless

1. Running fewer than six and reporting the spread anyway.
2. Passing `--seed` to the trainings, which would hide the variance being
   measured.
3. Changing the protocol part-way, which would make the spread a mixture of two
   distributions rather than one.
4. Treating a wide spread as a disappointing result. **A wide spread is the
   finding**, and it would retroactively weaken several task-2 claims — which is
   the point of measuring it.
5. Failing to confirm the six runs actually differ. Six distinct `md5sum` values
   are required (E24's failure mode).

## A seventh observation, labelled as such

E66's `full` control arm — **5.17 coins** — was produced by this exact protocol
and is a legitimate seventh draw from the same distribution. It will be reported
alongside the six, and **excluded from the pre-registered statistics**, because it
was observed before this registration was written.

## Power

Not applicable — nothing is being tested. But note that six observations
estimate a standard deviation poorly: the interval around an sd at n = 6 is wide.
The number will be used as an **order-of-magnitude threshold**, not as an exact
quantity, and any later test built on it will say so.
