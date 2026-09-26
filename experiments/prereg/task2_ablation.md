# Pre-registration — the remaining ablation arms on task 2

Registered **before** any run. Model LinQ, `experiments-v1` worktree.

## Question

E18 ablated seven feature groups in the **tournament** configuration and found
three nulls (`no_opp`, `no_danger`, `no_mask`), two real effects (`no_escape`,
`no_shaping`) and one unreadable row (`no_conj`).

E64 then re-ran **one** of those arms — `no_danger` — on task 2, because its
weight vector happened to survive the September cleanup. The tournament null of
**+0.27** became **−3.15 coins** on task 2, driven by an 83% collapse in bombing
with suicides unchanged at 0.00.

E63 and E65 established the mechanism: the four-player configuration masks
crate-opening value, because opened crates are a public good.

**Does the same masking apply to the other arms?**

## Hypothesis

Stated in advance, and it is directional:

> The arms E18 reported as **nulls** (`no_opp`, `no_mask`) are the ones at risk
> of hiding a solo effect. The arms E18 reported as **real** (`no_escape`,
> `no_shaping`) should remain real on task 2 — they produce pacifist agents that
> cannot score anywhere. `no_conj` is unpredicted.

A null that stays null on task 2 is **not** a failed experiment. It bounds the
generality of rule 16, which currently rests on a single arm.

## Design

- **Arms:** 6 — `full` (control), `escape`, `conj`, `opp`, `shaping`, `mask`
- **n:** 1 weight vector per arm. Stated plainly: **this is exploratory.**
- **Control:** trained fresh in this worktree under the identical protocol, *not*
  the stored `w_full` from the E18 era. A control from a different code state is
  not a control.
- **Training (E18's protocol, unchanged):** 400 rounds solo on `classic`, then
  300 rounds on `classic` vs 3× `peaceful_agent`. From scratch — each arm writes
  to its own `LQ_WEIGHTS` file, which does not exist beforehand.
- **Switch on during training *and* evaluation.** The feature is deleted, not
  disabled at test time.
- **Evaluation (E63's protocol):** solo `classic`, 100 rounds, learning off,
  **`--seed 1` for every arm** so all six face identical boards. This is a
  deliberate improvement on E64, which was unpaired.

## Primary metric

**Coins per round on task 2.** Fixed in advance.

## Decision rule — and why it is not a t-test

There is **one weight vector per arm**, so no test of significance is available
and none will be reported. E29 measured training spread at 1.68 margin over six
identical trainings; a single draw cannot resolve anything smaller.

Registered rule:

1. **A difference of ≥ 2.0 coins from the control counts as a signal**, and only
   as grounds for a powered follow-up — never as a measured effect size.
2. **A ≥ 50% change in bombs per round counts as a behavioural collapse**, which
   is categorical and survives n = 1. This is what carried E64, not its coin
   number.
3. **Anything smaller is "not resolved at n = 1"** and will be written that way.

No arm will be described as a null on task 2 on the strength of this design. The
most it can establish is that an arm **does** hide an effect.

## Secondaries

Crates per round, crates per bomb, bombs per round, suicides. Reported for
mechanism, claimed as effects never — see E54, where a positive-looking secondary
was refused three times.

## What would make this experiment worthless

1. Reporting a coin difference below 2.0 as an effect.
2. Comparing against `w_full` from the E18 era instead of the fresh control.
3. Failing to confirm the switch actually reached the agent. `setup()` logs
   `ablated=...` and warns `FEATURES ABLATED` — **check the log, and check the
   six weight files differ by `md5sum`** (E24's failure mode).
4. Running the shaping or mask arm with `LQ_ABLATE` still exported from the
   previous arm. Every command carries its own inline environment.

## Verification before the runs

- `ABLATION_GROUPS` contains `escape`, `conj`, `opp` — confirmed at
  `features.py:180`.
- `LQ_SHAPING` exists at `train.py:44`; `LQ_MASK_INVALID` at `callbacks.py:55`.
  Neither is an ablation group, which is why both are set separately.
- After all arms: `md5sum out_weights/*.npz` must show six distinct hashes.
