# Pre-registration — A3, do the conjunctions and the invalid-action mask cost task 2?

Registered **before** any run, 20 Sept, after E67 and because of it. Model LinQ,
`experiments-v1` worktree, GitHub codespace (Python 3.12.1, numpy 2.5.3).

## Question

E66 measured, at n = 1 per arm on task 2:

| arm | coins |
|---|---|
| control | 5.17 |
| `opp` | 6.50 |
| `mask` | 7.68 |
| `conj` | 8.74 |

All three ablated arms scored **above** the control. Both `conj` and `mask`
cleared E66's registered threshold of 2.0 coins and **both were refused**,
because a single seed could not be separated from training variance.

E67 then measured the control properly and found something that changes the
question: the distribution is **bimodal**. Three of six identical trainings score
exactly zero; the other three score 4.97, 5.25, 5.57.

**So all three E66 arms sit above every non-collapsed control run ever
measured.** That is the observation this experiment exists to test.

**Question: do the conjunction features and the invalid-action mask make the
agent worse on task 2?**

## Why the obvious design is invalid

A paired t-test on coins per round **cannot be used.** Half of all control runs
score exactly zero, so the sample is a mixture of two populations and a mean
difference describes neither. This is E18's `no_conj` lesson and E67's rule 18.

## Two primary outcomes, both fixed in advance

**P1 — collapse rate.** The fraction of trainings scoring zero coins.
Control: **3 of 6** (E67). Compared between arms by **Fisher's exact test,
two-sided, α = 0.05.**

**P2 — coins conditional on learning.** Mean coins over non-collapsed runs only.
Control: **5.26, sd 0.30, range 4.97–5.57** (E67, n = 3).

**Conditioning on non-collapse is a selection step, and it is declared here in
advance.** It is applied identically to every arm, and P1 reports the discarded
runs rather than hiding them. Reporting P2 without P1 would be a fishing
expedition.

## Decision rule

- **P1 is claimed** only if Fisher's exact test gives p < 0.05.
- **P2 is claimed** only if the arm's conditional mean falls **outside
  4.97–5.57**, the full observed range of every non-collapsed control run, *and*
  the arm has at least 4 non-collapsed runs.
- **Both must be reported whatever they show.** An arm that changes the collapse
  rate but not the conditional mean is a different finding from one that does the
  reverse, and neither may be presented as the other.

**The conditional sd of 0.30 rests on three points and is a weak estimate.** It
is not used as a test statistic. The observed control *range* is the comparator,
and "outside the range" means *interesting*, not *established*.

## Design

- **Arms:** 3 — `control`, `conj` (`LQ_ABLATE=conj`), `mask`
  (`LQ_MASK_INVALID=0`)
- **n = 12 per arm.** With a ~50% collapse rate this yields about 6 usable runs
  per arm. Eight would have yielded four, which is not enough to compare
  conditional means.
- **Control reuses E67's six runs and adds six more** (seeds 7–12), same
  protocol, same platform, same day. Declared, not silent.
- **Training:** 400 rounds solo on `classic`, then 300 vs 3× `peaceful_agent`,
  from scratch, each run writing its own `LQ_WEIGHTS` file. Identical to E66 and
  E67.
- **Evaluation:** solo `classic`, 100 rounds, learning off, `--seed 1`.
- **Switch on during training and evaluation.**

## Power, stated honestly

**P1:** at n = 12 per arm, Fisher's exact test comparing 50% against an
alternative can detect only a large shift — roughly 50% vs 8% or lower. A null on
P1 means "no large change in collapse rate detected", never "no change".

**P2:** with ~6 usable runs per arm against a control sd estimated from 3 points,
this resolves differences of order 1 coin, not 0.2. Both E66 gaps (+2.42, +3.48)
are far larger than that; if they are real, this design sees them.

## What would make this experiment worthless

1. Reporting P2 without P1.
2. Changing the conditioning rule after seeing which runs collapsed.
3. Using E66's single-seed gaps to set the threshold — E47's error, and the
   reason the threshold here comes from E67's control instead.
4. Treating "outside the control range" as a measured effect size. It is a
   trigger for a larger study, and this project has refused four
   positive-looking results already for exactly this reason.
5. Failing to verify the arms differ: 30 distinct `md5sum` values required.

## Verification before the runs

- `ABLATION_GROUPS` contains `conj` — `features.py:180`.
- `LQ_MASK_INVALID` — `callbacks.py:55`.
- Five-round smoke test showing `ablated=['conj']` in the agent log before the
  full launch, as in E66.

## A prediction, recorded so it can be wrong

If the conjunctions and the mask are genuinely costing task 2, the most likely
mechanism given E67 is **a change in the collapse rate, not a shift of the
conditional mean** — because the bimodality is a discovery failure (A2), and
these features act on exploration and action selection rather than on scoring.

**P1 moving while P2 stays flat is therefore the predicted outcome. If instead P2
moves and P1 does not, the mechanism above is wrong** and the write-up must say
so rather than reinterpreting.
