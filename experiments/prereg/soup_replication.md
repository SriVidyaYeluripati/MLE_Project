# Pre-registration — B1, does the weight soup reproduce?

Registered **before** any run, 20 Sept. Model LinQ, `experiments-v1` worktree,
codespace (Python 3.12.1, numpy 2.5.3).

## The claim under test

E31 is **already in the report**. It says:

> For a linear Q this is well defined and free: `w` and `c·w` induce the
> identical greedy policy, so the soup normalises each vector to unit length
> first. **The soup lands on the mean of the distribution, not above it.** It
> does not reach the best seed's +1.06 and was never going to — averaging is a
> variance operation, not a performance one. **What it removes is the −0.62
> tail.** A tournament is a single draw, and a recipe whose outcomes run from
> −0.62 to +1.06 is a lottery ticket. **Report it as risk removal, not as an
> improvement.**

That rests on **one** set of six seeds (E29's). This experiment asks whether a
second, independent set of six reproduces it.

**A wrong claim in the report costs more than a missing one.** That is the whole
reason this run exists.

## Protocol — E29's, unchanged

Deviating here would make the comparison meaningless, so every parameter is
copied:

- **Warm start** from the worktree's committed `weights.npz`. E29's seeds did not
  train from scratch, which is why none of them collapsed. The ~58% collapse rate
  measured in E67 and E68 applies to *from-scratch* training and is **not**
  expected here.
- **600 rounds vs 3× `peaceful_agent`** on `classic`, differing only in world
  seed.
- **Evaluation: 300 rounds vs 3× `rule_based_agent`**, learning off, `--seed 1`
  for every arm so all seven face identical boards.
- **n = 6 seeds**, plus the soup built from them = 7 evaluations.

## Primary metric

**Margin** = our score per round minus the best `rule_based_agent`'s score per
round on the same boards. E29's unit, and E33's reason for preferring it: raw
score moves with board difficulty by more than the effects being measured.

## The soup

Each of the six weight vectors is scaled to unit L2 norm, then averaged
elementwise. Unit-normalising first is not cosmetic: `w` and `c·w` induce the
same greedy policy, so an unnormalised mean would silently weight the
longest vector most.

## Decision rule, fixed in advance

E31 makes two separate claims. They are tested separately.

**C1 — "the soup lands on the mean, not above it."**
Reproduced if the soup's margin falls **within the six seeds' range**, and
specifically does **not exceed the best seed**.
**Falsified** if the soup beats every individual seed — which would make it an
improvement, not a variance operation, and would mean E31 understates it.

**C2 — "what it removes is the tail."**
Reproduced if the soup's margin is **above the worst seed's**.
**Falsified** if the soup lands at or below the worst seed.

**Both are reported whatever they show.** C1 and C2 can fail independently, and
a mixed outcome is a real result, not a mess to be tidied.

## Power, stated honestly

Six seeds estimate a spread poorly and **one** soup is a single draw. E29 put
evaluation variance at ~0.13 margin against training variance of 1.68, so a
single evaluation of the soup is a reliable reading *of that soup* — but a
different six seeds would give a different soup.

This experiment can therefore show that E31's pattern **does or does not appear
again**. It cannot estimate how often it appears. A confirmation licenses the
sentence already in the report; it does not strengthen it.

## What would make this experiment worthless

1. Training from scratch instead of warm-starting. Different protocol, different
   question, and E67's collapse rate would dominate everything.
2. Normalising after averaging instead of before.
3. Reporting C1 or C2 alone.
4. Quoting the soup's margin as an improvement. E31 explicitly says it is not
   one, and the point of this run is to check E31, not to relitigate it.
5. Failing to verify the six differ — six distinct `md5sum` values required, plus
   a seventh for the soup.

## Prediction, recorded so it can be wrong

E31's mechanism is that averaging reduces variance without raising the mean. If
that is right, the soup should land **near the middle of the six and above the
worst**, and both C1 and C2 reproduce.

**If the soup instead beats all six, E31's "risk removal, not improvement"
framing is wrong** and the report sentence needs rewriting — in the agent's
favour, which is exactly the kind of correction that is easy to accept and
therefore easy to get wrong.
