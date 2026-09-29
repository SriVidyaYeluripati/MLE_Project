# Experiments

Scripts, pre-registrations, and trained weight vectors for every run behind
the report.

    prereg/                    pre-registration for each run, written before the run
    model_linearQ/              scripts and weight vectors for the linear agent
    model_linearQ/weights/      one folder per experiment, one file per training run

Raw results for every model are in `results/`, one folder per model, plus
`results/cross_model/` for the joint evaluations.

## How we run things here


1. One weight file per training run. We never overwrite a run in a loop.
2. Every arm of an experiment sees the same seed, so they play the same boards.
3. The training run is the unit of replication — not the evaluation games we
   run afterward on a fixed model.
4. The metric, the number of seeds, and the decision rule are fixed before the
   run starts, written down in `prereg/`. We don't pick the metric after
   looking at the numbers.
5. We record a result whether the change worked or not. A null result is still
   a result.

## Still missing: GBT and DQN

Only LinearQ has a scripts-and-weights folder here. Draft notes below —
delete each once it's actioned.

> **Pratik — draft note:** your GBT results are in `results/model_gbt/`, but
> the training scripts and weight files behind them aren't in the repo yet.
> Could you push an `experiments/model_gbt/` folder, same shape as
> `experiments/model_linearQ/` — scripts at the top level, a `weights/`
> subfolder with one file per training run, not one file overwritten in a
> loop? And a pre-registration in `prereg/` for whichever result becomes your
> headline number in the report (name it `prereg/gbt_<topic>.md`, the
> existing files are a template). Also, two things from the report checklist
> while you're in there: who first spotted the symmetry sign error and when
> (for §1's chronology), and your full name for the section header.

> **Shayan — draft note:** same ask for DQN — an `experiments/model_dqn/`
> folder with your training script(s) and a `weights/` subfolder. If "one
> weight file per training run" doesn't map onto how you checkpoint DQN
> (e.g. saving periodically within one run rather than one file per run),
> say so directly rather than forcing it — and then say in your report
> section what the actual unit of replication is, so rule 3 above doesn't
> quietly misdescribe your evaluation. Also from the checklist: when did DQN
> leave the shared 28-feature contract, and what hardware did you train on
> (needed for §3, project planning)?

Once both are in, this section comes out and the folder list above gets
`model_gbt/` and `model_dqn/` rows.

## Naming note

ForestQ was still called `model_a` while the runs in `results/forestq/` were
made, so the agent name inside those result files is `model_a`. We left them
as raw framework output rather than editing them after the fact.