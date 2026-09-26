# Experiments

Scripts, pre-registrations and trained weight vectors for the experiments
reported in the project report.

    prereg/            pre-registrations, each written before the run it describes
    model_linearQ/     scripts and weight vectors for the linear agent
    model_linearQ/weights/   one folder per experiment, one file per training run

Raw results are in `results/`, one folder per model.

## Conventions

1. One weight file per training run. Nothing is overwritten in a loop.
2. All arms of an experiment use the same seed, so they face the same boards.
3. The unit of replication is the training run, not the evaluation.
4. The measure, the number of seeds and the decision rule are fixed before the
   run, in `prereg/`.
5. Results are recorded whether or not the change worked.

## Naming note

ForestQ was called `model_a` while the runs in `results/forestq/` were made, so
the agent name inside those result files is `model_a`. They are raw framework
output and have not been edited.
