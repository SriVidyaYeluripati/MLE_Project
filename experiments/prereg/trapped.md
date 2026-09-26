# Pre-registration — retraining on a repaired `opp_trapped`

Written **before** the runs, as with `PREREG_bombing.md`.

## The bug

`_danger_view` clears every opponent's tile from `free`, which is correct for
our own escape planning — a body in a corridor really does block us.
`survivable_map` requires `free[tile]`, so `surv[0][o]` is **False at every
opponent's own tile, unconditionally**, on an empty board with no bombs.

`opp_trapped` asks `o in hit and not surv[o]`. The second clause is always
true. So the feature has only ever computed `any(o in hit)` — which *is*
`opp_in_blast`, the feature stage 4 built it to replace.

Evidence:

- On a wide-open board with no bombs, `surv[0]` at an opponent's own tile is
  False and at the empty tile beside it is True.
- In 76,000 steps of real play, `opp_in_blast` and `opp_trapped` fired on
  exactly the same **443 of 443** steps.
- All four kill weights sit at exactly **+0.0427** — a learner cannot split
  credit between columns it cannot distinguish.

## The fix, and why it is believable

`surv_opp` is a second survivability map built on `field == 0`, so agents are
not obstacles to *their own* escape. `LQ_FIX_TRAPPED=1` selects it.

Validation against ground truth, 200 rounds vs 3× `coin_collector`:

| | bombs covering an opponent | claimed as trapping |
|---|---|---|
| broken | 443 | 443 (100%) |
| **fixed** | 502 | **23 (5%)** |

and the agent's **actual** kill conversion is **~22 kills from 435 covering
bombs = 5.1%**. The repaired feature fires at the rate that kills actually
happen. The broken one fired on everything.

## What this experiment does and does not test

The agent already takes **21 of 23** genuine trap opportunities. It is not
declining kills. So the fix cannot help by making it take chances it already
takes — the binding constraint is that only 0.115 genuine traps arise per
round, which is exactly its kill rate.

The hypothesis is therefore **not** "the agent will convert better". It is:

> a feature that is constant-on-cover provides no gradient for *positioning*.
> Retrained on a feature that discriminates, the learner may seek trap
> geometry rather than treating all coverage alike, and so create more
> opportunities.

That is a real possibility and it is also quite likely to be a null. Stated
now so the null is not reinterpreted afterwards.

## Arms and metric — fixed now

Two arms, **six seeds each**, seeds 3001–3006. Warm start from `weights.npz`,
600 rounds on `classic` vs three `peaceful_agent`s, evaluated 300 rounds vs
three `coin_collector_agent`s.

| tag | switch |
|---|---|
| `tbase` | none |
| `tfix` | `LQ_FIX_TRAPPED=1` at train **and** eval |

**Primary metric: kills per round.** That is the channel under test and the
quantity the mechanism acts on. **Secondary: score.** Margin is reported last
and with its spread, because it is the noisiest unit available (finding 20).

**Decision rule:** paired by seed. An improvement requires the paired mean
difference in kills to be positive with |t| > 2.57 (p<.05, n=6), *and* score
not to fall. Anything else is a null.

## Power, stated in advance

At n=6 this test has ~37% power for a one-standard-deviation effect. **It can
only find a large effect.** A null here bounds the effect loosely and should
be reported as "not detected at n=6", not as "no effect".

## What would make this worthless

- Reporting the best seed rather than the paired mean.
- Switching to score or margin because kills says nothing.
- Treating a null as confirmation that the bug does not matter — the bug is
  established independently of whether repairing it raises the score.
