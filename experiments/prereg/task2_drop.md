# Pre-registration — drop-time crate reward on task 2

Registered **before** any run. Model LinQ, `agent_code/model_linearQ`.

## Question

E37 tested the drop-time crate reward (`LQ_DROP_REWARD`) in the tournament
configuration and found it null on the pre-registered primary: crates per bomb
+0.49, better in 4 of 4 seeds, but t ≈ 1.5 at n = 4 on a rule that required more.
It is the only intervention in this project that has never been shown *not* to
work.

E63 then established that the tournament and task 2 are different problems.

## Hypothesis and why it is not a fishing expedition

E37 and E40 explain the tournament null by **crate-opening being a public good**:
the coins released are collected by opponents at least as well as by this agent.

That explanation makes a prediction about a case it was not derived from.
**Solo there are no opponents, so the public-good mechanism cannot operate**, and
an intervention that raises crates per bomb should convert directly into coins.

E63 sized the effect available: on task 2 the shipped agent clears 59% of crates
at 1.34 crates/bomb, against the step-3 agent's 92% at 2.57 and `rule_based`'s
96% at 3.09. The gap between 1.34 and 2.57 is worth **2.68 coins**.

**Prediction: the drop-time crate reward raises coins on task 2, despite being
null in the tournament.**

## Design

- **Arms:** 2 — baseline and `LQ_DROP_REWARD=1`
- **n:** 10 seeds, fixed before any data exists
- **Pairing:** by seed. Both arms start from the same warm-start weights and use
  the same training world seed.
- **Training:** 600 rounds on `classic` vs 3× `peaceful_agent`
- **Evaluation:** solo on `classic`, 100 rounds, learning off, **switch off in
  both arms** — the reward is a training-time change and evaluation is on the
  unshaped task, as everywhere else in this project.

## Primary metric

**Coins per round on task 2.** Fixed in advance.

## Decision rule

One-sided paired t-test, df = 9. **t > 1.833** to claim the effect.

## Secondaries

Crates per round, crates per bomb, bombs per round, steps. **Reported, and
claimed only if the primary passes.** This project has refused a positive-looking
secondary three times (E54) and that discipline is the reason the record is clean.

## Power, computed honestly

Do **not** use E37's spread — it is an exploratory estimate and E47 established
that such estimates are biased low by selection. Using E29's control condition
instead (six identical trainings), the coins sd is of order 0.6–0.75 in the
tournament setting; task 2 is solo and less noisy, but no control has been
measured for it, so treat the power as **unknown** and read a null as "not
detected", not "absent".

## What would make this experiment worthless

1. Switching to crates or crates-per-bomb because coins says nothing.
2. Reporting the best seeds.
3. Adding an arm after seeing the results.
4. Failing to verify that the independent variable actually changed — check that
   `weights.npz` differs between arms (E24's failure mode).

## Verification before the runs

- `grep -n LQ_DROP_REWARD agent_code/model_linearQ/train.py` confirms the switch
  exists at tag `experiments-v1` (line 102).
- After the first seed, `md5sum` the two arms' weight files and confirm they
  differ.
