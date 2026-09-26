# Pre-registration — A1/A2, the drop-time crate reward under the shipped feature set

Registered **before** any run, 20 Sept. Model LinQ, **shipped code** (`Vidya`
branch, 33 features), codespace, Python 3.12.1, numpy 2.5.3.

**This is the only experiment in this project that could change the submitted
agent.** The deadline is 21 Sept 21:00.

## Question

E37 tested the drop-time crate reward in the tournament at n = 4 and got t ≈ 1.5
— the only intervention in this project never shown not to work. E65 then tested
it on task 2 with ten paired seeds and it **passed**: +4.66 coins, 10 of 10
seeds, t = +5.41 against a threshold fixed in advance.

It was not shipped, because the E65 weights are 40-column and the submitted agent
is 33-column. That was an incompatibility, not a judgement.

**Two questions remain, and one set of trainings answers both:**

- **A1** — does E65's task-2 gain survive under the **shipped 33-feature** code?
- **A2** — does the drop reward **cost tournament performance**? Never measured.

## The code change, and what it does not touch

Three lines added to `train.py` only:

```python
DROP_REWARD = os.environ.get('LQ_DROP_REWARD', '0') != '0'
if DROP_REWARD:
    REWARDS_EXTRA[e.CRATE_DESTROYED] = 0.0
```

and inside `reward_from`:

```python
    if DROP_REWARD and e.BOMB_DROPPED in events and old_ctx is not None:
        x, y = old_state['self'][3]
        r += 0.02 * int(old_ctx['bomb_value'][x, y])
```

**`features.py` and `callbacks.py` are not modified.** Those are the files loaded
during official games; `train.py` is imported only in training mode, so this
change cannot affect tournament behaviour. The only thing that could change in
the submission is `weights.npz`.

The change is made in a **worktree**, not in the submitted tree.

## Design

- **Arms:** 2 — `base` and `drop` (`LQ_DROP_REWARD=1`)
- **n = 6 paired seeds**, fixed before any data exists
- **Warm start** from the shipped `weights.npz` for both arms — identical
  starting point, same world seed per pair
- **Training:** 600 rounds on `classic` vs 3× `peaceful_agent`, E29/E65's
  protocol
- **Two evaluations per vector**, both at `--seed 1` so every arm faces identical
  boards:
  - **Tournament:** 300 rounds vs 3× `rule_based_agent` → **A2**
  - **Task 2:** 100 rounds solo → **A1**
- **Reference:** the current shipped `weights.npz` evaluated in both
  configurations on the same boards. Without the incumbent's number on these
  boards, nothing here is decidable.
- Reward switch **on during training only**. Evaluation is on the unshaped task
  in both arms, as everywhere in this project.

## Metrics

- **A2 (primary, decision-relevant):** tournament **margin** = our score per
  round minus the best `rule_based_agent` on the same boards. E29's unit; E33's
  reason for preferring it over raw score.
- **A1 (secondary):** coins per round on task 2.

## The swap rule — fixed now, before any number exists

**A new vector is submitted only if all five hold. Any single failure means the
zip is untouched.**

1. **No tournament loss.** The drop arm's mean margin is **≥** the base arm's
   mean margin across the six paired seeds.
2. **Task 2 improves.** The drop arm's mean task-2 coins exceed the base arm's,
   replicating E65 under the shipped features.
3. **The candidate is the soup, not a seed.** The submitted vector is the
   **unit-normalised mean of all six drop vectors** — E31's construction,
   re-confirmed this morning by E69. **Selecting the best of six is forbidden**,
   and E68 is this project's own demonstration of why.
4. **The soup beats the incumbent.** The soup's tournament margin on the same 300
   boards is **≥** the current shipped `weights.npz`. This is the real gate: a
   challenger must beat the agent that exists, not merely its own control arm.
5. **The feature vector is unchanged.** The new weights must be 33-column and
   load without the version guard firing.

**Default is not to swap.** The incumbent has weeks of testing; a challenger will
have forty minutes. Where the rule is ambiguous, the zip stays.

## Power, stated honestly

Six paired seeds against E29's training spread of 1.68 margin resolves
differences of roughly 0.7 margin, not 0.2. **A null on A2 means "no large
tournament cost detected", not "no cost".** Rule 1 is deliberately written as
non-inferiority rather than as a significance test, because at n = 6 a
significance test on the tournament metric would fail even for a real effect, and
because "does not lose" is the question that matters for a swap decision.

## What would make this experiment worthless

1. Selecting the best drop seed to ship. Forbidden by rule 3.
2. Skipping the incumbent reference evaluation, leaving rule 4 untestable.
3. Modifying `features.py` or `callbacks.py`. Neither is needed.
4. Evaluating with the reward switch still on — training-time change only.
5. Swapping on task 2 alone. Task 2 is one of four tasks, and the tournament is
   what the submission is for.
6. Failing to verify the arms differ: 12 distinct `md5sum` values.

## Prediction, recorded so it can be wrong

E37 and E40 explained the tournament null by crate-opening being a **public
good** — opponents collect the released coins too. That predicts **no tournament
gain and no large tournament loss**: the reward changes when crates are paid for,
not whether the agent survives or kills.

**So the predicted outcome is rule 1 passing narrowly, rule 2 passing clearly,
and the swap decision resting on rule 4.** If instead the drop arm loses
materially in the tournament, the public-good explanation is incomplete and the
write-up says so rather than reinterpreting.
