#!/usr/bin/env python3
"""
Write results/model_linq/summary/*.json - one file per experiment.

    python tools/model_linq/build_summaries.py

Design, pre-registration, aggregate numbers and verdict for every experiment run
on Model LinQ, transcribed from the build log. Raw per-run results stay in the
sibling folders named by each entry's "raw_runs" field. Re-runnable: it
overwrites whatever is there. Figures come from make_figures.py, which reads
the files this writes.
"""

import json
import os

OUT = os.path.join('results', 'model_linq', 'summary')

SUMMARIES = json.loads(r"""
{
 "ablations.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "ablations",
  "title": "Feature-group ablations",
  "section": "build log 5",
  "question": "Which feature groups actually carry the policy?",
  "design": {
   "arms": [
    "full",
    "no_escape",
    "no_shaping",
    "no_conj",
    "no_opp",
    "no_danger",
    "no_mask"
   ],
   "n": 3,
   "repeats_per_arm": 3,
   "total_trainings": 21,
   "paired": false,
   "primary_metric": "margin",
   "protocol": "trained from scratch: 400 rounds solo on classic, then 300 vs 3x peaceful_agent; evaluated over 200 tournament rounds with learning off",
   "decision_rule": "baseline spread +-0.19 over three runs, so |delta| > 0.38 counts"
  },
  "raw_runs": "results/model_linq/ablation/",
  "arms": [
   {
    "arm": "full",
    "margin": 0.4,
    "margin_range": [
     0.18,
     0.52
    ],
    "delta": null,
    "score": 3.27,
    "coins": 2.67,
    "kills": 0.12,
    "crates": 27.56,
    "suicides": 0.28,
    "invalid": 0.28,
    "alive": 335.3
   },
   {
    "arm": "no_escape",
    "margin": -4.9,
    "margin_range": [
     -5.22,
     -4.63
    ],
    "delta": -5.3,
    "score": 0.02,
    "coins": 0.02,
    "kills": 0.0,
    "crates": 0.51,
    "suicides": 0.18,
    "invalid": 0.74,
    "alive": 187.6
   },
   {
    "arm": "no_shaping",
    "margin": -3.89,
    "margin_range": [
     -4.0,
     -3.67
    ],
    "delta": -4.29,
    "score": 0.01,
    "coins": 0.01,
    "kills": 0.0,
    "crates": 0.0,
    "suicides": 0.01,
    "invalid": 2.11,
    "alive": 350.7
   },
   {
    "arm": "no_conj",
    "margin": -1.05,
    "margin_range": [
     -4.27,
     1.0
    ],
    "delta": -1.45,
    "score": 2.23,
    "coins": 1.92,
    "kills": 0.06,
    "crates": 18.79,
    "suicides": 0.22,
    "invalid": 0.44,
    "alive": 314.5
   },
   {
    "arm": "no_opp",
    "margin": 0.72,
    "margin_range": [
     0.28,
     1.02
    ],
    "delta": 0.32,
    "score": 3.61,
    "coins": 2.9,
    "kills": 0.14,
    "crates": 30.69,
    "suicides": 0.44,
    "invalid": 1.02,
    "alive": 286.8
   },
   {
    "arm": "no_danger",
    "margin": 0.67,
    "margin_range": [
     0.15,
     1.31
    ],
    "delta": 0.27,
    "score": 3.54,
    "coins": 2.69,
    "kills": 0.17,
    "crates": 30.89,
    "suicides": 0.3,
    "invalid": 0.84,
    "alive": 331.9
   },
   {
    "arm": "no_mask",
    "margin": 0.3,
    "margin_range": [
     -0.17,
     0.79
    ],
    "delta": -0.1,
    "score": 3.17,
    "coins": 2.55,
    "kills": 0.12,
    "crates": 32.39,
    "suicides": 0.2,
    "invalid": 0.54,
    "alive": 355.9
   }
  ],
  "verdict": "two groups load-bearing",
  "caveat": "The trapped and conj rows are contaminated: x_bomb_opp is an identical column to opp_in_blast and x_bomb_trapped to opp_trapped (see kill_channel), so those ablations do not measure what their names say. escape, shaping, danger, coins, crates and opp are unaffected.",
  "conclusion": "Survival (no_escape, -5.30) and potential-based shaping (no_shaping, -4.29) are the only groups whose removal clears the noise floor. Opponent and danger features can be deleted with no measurable loss."
 },
 "action_filters.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "action_filters",
  "title": "Action filters from the Pommerman literature",
  "section": "build log 8",
  "question": "Skynet (Gao et al. 2019) and Kartal et al. (2019) report an action filter as the component that makes model-free RL viable in Bomberman. Does it help here?",
  "design": {
   "arms": [
    "shipped",
    "+safety filter",
    "+bomb filter",
    "+both"
   ],
   "evaluation": "200 rounds, learning off",
   "note": "play-time only - no retraining, no weight change"
  },
  "arms": [
   {
    "arm": "shipped",
    "margin_vs_rule_based": [
     0.35,
     0.44
    ],
    "margin_vs_coin_collector": [
     -0.09,
     -0.17
    ],
    "suicides": 0.46,
    "bombs": 37.5,
    "crates": 22.9,
    "crates_per_bomb": 0.61
   },
   {
    "arm": "+safety filter",
    "margin_vs_rule_based": [
     0.45,
     0.33
    ],
    "margin_vs_coin_collector": [
     0.11,
     0.04
    ],
    "suicides": 0.45,
    "bombs": 38.3,
    "crates": 23.5,
    "crates_per_bomb": 0.61
   },
   {
    "arm": "+bomb filter",
    "margin_vs_rule_based": null,
    "margin_vs_coin_collector": null,
    "suicides": 0.18,
    "bombs": 20.7,
    "crates": 25.8,
    "crates_per_bomb": 1.25
   },
   {
    "arm": "+both",
    "margin_vs_rule_based": [
     0.29,
     0.46
    ],
    "margin_vs_coin_collector": [
     -0.0,
     -0.08
    ],
    "suicides": 0.14,
    "bombs": 20.8,
    "crates": 26.1,
    "crates_per_bomb": 1.26
   }
  ],
  "verdict": "null",
  "conclusion": "The safety filter has nothing left to remove: no_escape is already the second-largest learned weight, so the exact survival information the filter enforces is already inside the feature vector. A published component is a hypothesis about the agent it was built for, not a free improvement."
 },
 "bc_ceiling.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "bc_ceiling",
  "title": "The ceiling diagnostic: are the features or the learner the constraint?",
  "section": "run output (post-dates the build log)",
  "source_note": "Numbers from tools/model_linq/bc_probe.py and tools/model_linq/dagger.py. Raw runs in results/model_linq/dagger/.",
  "question": "Every intervention has been a null. Is the binding constraint the feature set, the linear function class, or the RL algorithm?",
  "design": {
   "method": "behavioural cloning of rule_based_agent onto LinQ's exact functional form (conditional logit on the action scores), then DAgger",
   "why": "if the best possible model on these features cannot imitate rule_based, no weight vector on these features plays like rule_based, and the features are the ceiling"
  },
  "raw_runs": "results/model_linq/dagger/",
  "agreement_with_rule_based": [
   {
    "model": "shipped LinQ weights",
    "pct": 40.6
   },
   {
    "model": "best linear model on the same features",
    "pct": 79.7
   },
   {
    "model": "best linear model + 128 random tanh features",
    "pct": 79.8
   }
  ],
  "play_performance": [
   {
    "model": "shipped LinQ (RL)",
    "score": 3.16
   },
   {
    "model": "behavioural clone",
    "score": 2.15
   },
   {
    "model": "DAgger round 1",
    "score": 2.08
   },
   {
    "model": "DAgger round 2",
    "score": 1.21
   },
   {
    "model": "DAgger round 3",
    "score": 0.58
   },
   {
    "model": "DAgger round 4",
    "score": 0.58
   }
  ],
  "dagger_accuracy_final": 74.5,
  "verdict": "the features are the ceiling",
  "conclusion": "The best possible linear model on these 36 features agrees with rule_based only 79.7% of the time, and adding 128 non-linear features buys 0.1 points of that - so the remaining 20% is not learnable from these numbers, because the same feature vector maps to different rule_based actions. The clone also plays worse than the RL agent (2.15 vs 3.16), which means the RL already extracts more from these features than imitation does. This retired the pre-registered random-feature experiment before it was run."
 },
 "bombing.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "bombing",
  "title": "Bombing efficiency: five pre-registered arms",
  "section": "build log 11",
  "question": "The agent destroys 0.43 crates per bomb against coin_collector 2.89. Can that be fixed?",
  "preregistration": "experiments/model_linq/PREREG_bombing.md",
  "design": {
   "arms": [
    "drop-time crate reward",
    "selective crate targets",
    "both",
    "trained inside the bomb filter",
    "filter bolted on at play time"
   ],
   "n": 4,
   "paired": true,
   "primary_metric": "margin",
   "fixed_in_advance": true
  },
  "raw_runs": "results/model_linq/bomb/",
  "diagnosis": {
   "mechanism": "CRATE_DESTROYED pays 0.02 per crate but arrives 4 steps after the bomb (BOMB_TIMER=4); the trace decays by gamma*lambda = 0.76 per step, so the bombing action retains 0.76^4 = 0.334 of it",
   "predictions": [
    {
     "feature": "crates_hit_1",
     "predicted": 0.0067,
     "measured": 0.0073
    },
    {
     "feature": "crates_hit_3p",
     "predicted": 0.0267,
     "measured": 0.0274
    },
    {
     "feature": "is_bomb",
     "predicted": "paid immediately",
     "measured": 0.0348
    }
   ]
  },
  "arms": [
   {
    "arm": "drop-time crate reward",
    "crates_per_bomb": {
     "mean": 0.49,
     "better_in": "4/4",
     "t": 1.5
    },
    "coins": {
     "mean": 0.28,
     "better_in": "3/4"
    },
    "margin": {
     "mean": 0.39,
     "better_in": "3/4"
    }
   },
   {
    "arm": "selective crate targets",
    "crates_per_bomb": {
     "mean": 0.39,
     "better_in": "3/4"
    },
    "coins": {
     "mean": 0.05,
     "better_in": "2/4"
    },
    "margin": {
     "mean": -0.06,
     "better_in": "1/4"
    }
   },
   {
    "arm": "both",
    "crates_per_bomb": {
     "mean": 0.13,
     "better_in": "3/4"
    },
    "coins": {
     "mean": -0.21,
     "better_in": "1/4"
    },
    "margin": {
     "mean": -0.6,
     "better_in": "1/4"
    }
   },
   {
    "arm": "trained inside the bomb filter",
    "crates_per_bomb": {
     "mean": 1.35,
     "better_in": "4/4",
     "t": 8.7
    },
    "coins": {
     "mean": -0.08,
     "better_in": "2/4"
    },
    "margin": {
     "mean": -0.08,
     "better_in": "2/4"
    }
   },
   {
    "arm": "filter bolted on at play time",
    "crates_per_bomb": {
     "mean": 1.31,
     "better_in": "4/4",
     "t": 39.6
    },
    "coins": {
     "mean": 0.05,
     "better_in": "2/4"
    },
    "margin": {
     "mean": -0.11,
     "better_in": "3/4"
    }
   }
  ],
  "verdict": "five arms, five nulls on the primary",
  "conclusion": "Every arm moved the mechanism at overwhelming certainty (crates per bomb up to t=39.6) and none moved the score. In classic the coins are sealed inside the crates, so opening them efficiently is a public good the opponents collect at least as well."
 },
 "cross_model.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "cross_model",
  "title": "Model LinQ against the team's other models",
  "section": "build log 6",
  "question": "The project asks for two or more models and the submission of the best. Which is best?",
  "design": {
   "yardstick": "each model alone vs 3x rule_based_agent, classic, 200 rounds",
   "note": "a common reference is what makes separately developed models comparable: raw score drifts with board difficulty by more than the differences being claimed"
  },
  "raw_runs": "results/model_linq/cross_model/",
  "yardstick": [
   {
    "model": "model_linearQ",
    "owner": "Vidya",
    "score": 3.39,
    "margin": 0.53,
    "coins": 2.69,
    "crates": 23.39,
    "bombs": 27
   },
   {
    "model": "model_a",
    "owner": "Vidya",
    "score": 1.21,
    "margin": -2.32,
    "coins": 0.69,
    "crates": 7.87,
    "bombs": 23.07
   },
   {
    "model": "model_gbt",
    "owner": "Pratik",
    "score": 0.24,
    "margin": -4.97,
    "coins": 0.24,
    "crates": 0.0,
    "bombs": 0.0
   },
   {
    "model": "model_dqn",
    "owner": "Shayan",
    "score": 0.03,
    "margin": -5.28,
    "coins": 0.03,
    "crates": 0.0,
    "bombs": 0.0
   }
  ],
  "head_to_head": {
   "setup": "both models plus 2x rule_based_agent, 150 rounds",
   "rows": [
    {
     "agent": "model_linearQ",
     "score": 3.84,
     "margin": 0.54,
     "crates": 27.14
    },
    {
     "agent": "rule_based_agent_0",
     "score": 3.3,
     "margin": 0.0,
     "crates": 42.46
    },
    {
     "agent": "rule_based_agent_1",
     "score": 2.97,
     "margin": -0.33,
     "crates": 40.05
    },
    {
     "agent": "model_a",
     "score": 1.03,
     "margin": -2.27,
     "crates": 8.42
    }
   ]
  },
  "all_four_classic": {
   "setup": "all four team models in one game, classic, 200 rounds",
   "rows": [
    {
     "agent": "model_linearQ",
     "score": 6.89,
     "coins": 4.59,
     "kills": 0.46,
     "crates": 62.65,
     "bombs": 52.92,
     "suicides": 0.01,
     "alive": 397.8
    },
    {
     "agent": "model_a",
     "score": 0.23,
     "coins": 0.17,
     "kills": 0.01,
     "crates": 1.95,
     "bombs": 6.83,
     "suicides": 0.14,
     "alive": 370.2
    },
    {
     "agent": "model_gbt",
     "score": 0.1,
     "coins": 0.1,
     "kills": 0.0,
     "crates": 0.0,
     "bombs": 0.0,
     "suicides": 0.0,
     "alive": 335.9
    },
    {
     "agent": "model_dqn",
     "score": 0.02,
     "coins": 0.02,
     "kills": 0.0,
     "crates": 0.0,
     "bombs": 0.0,
     "suicides": 0.0,
     "alive": 375.0
    }
   ]
  },
  "all_four_coinheaven": {
   "setup": "all four on coin-heaven, 50 rounds, 50 coins shared",
   "rows": [
    {
     "agent": "model_linearQ",
     "score": 21.74,
     "coins": 18.04,
     "kills": 0.74
    },
    {
     "agent": "model_a",
     "score": 17.64,
     "coins": 15.74,
     "kills": 0.38
    },
    {
     "agent": "model_gbt",
     "score": 14.62,
     "coins": 14.62,
     "kills": 0.0
    }
   ]
  },
  "model_a_repair": [
   {
    "version": "as inherited",
    "margin": -2.8,
    "bombs": 0.22,
    "crates": 0.25,
    "suicides": 0.06,
    "alive": 325.9
   },
   {
    "version": "+ three bug fixes",
    "margin": -5.07,
    "bombs": 0.03,
    "crates": 0.02,
    "suicides": 0.0,
    "alive": 119.8
   },
   {
    "version": "+ reward and exploration fix",
    "margin": -2.32,
    "bombs": 23.07,
    "crates": 7.87,
    "suicides": 0.59,
    "alive": 238.4
   }
  ],
  "verdict": "Model LinQ is the submitted agent",
  "conclusion": "The gap of 2.85 margin between LinQ and Model A is 7.5x the +-0.38 threshold from the ablation suite, so unlike most differences in this project it is not a question of noise. The two independent measurements of LinQ agree to within 0.01 (+0.53 and +0.54). Model GBT and Model DQN, built separately by two teammates on the same feature contract, also score approximately zero on classic and neither has potential-based shaping - which is exactly what the no_shaping ablation predicts."
 },
 "deployment_temperature.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "deployment_temperature",
  "title": "Deployment exploration: does a published epsilon transfer?",
  "section": "build log 3 (finding 25)",
  "question": "chbridges (Heidelberg WS20/21) score better at a deployment epsilon of 0.25 than 0.05, because pure greedy oscillates. Does that transfer?",
  "design": {
   "arms": [
    "tau=0.00",
    "tau=0.05",
    "tau=0.10",
    "tau=0.15",
    "tau=0.40"
   ],
   "evaluation": "300 rounds vs 3x rule_based_agent",
   "note": "play-time only - no retraining, so no training noise",
   "evaluation_noise": 0.13
  },
  "arms": [
   {
    "arm": "tau=0.00",
    "margin": 0.49
   },
   {
    "arm": "tau=0.05",
    "margin": 0.56
   },
   {
    "arm": "tau=0.10",
    "margin": 0.45
   },
   {
    "arm": "tau=0.15",
    "margin": 0.32
   },
   {
    "arm": "tau=0.40",
    "margin": -0.63
   }
  ],
  "verdict": "null (greedy retained)",
  "conclusion": "The first four are within evaluation noise of each other and the last is clearly worse. The oscillation their agent suffered is already prevented here by the invalid-action mask and is_wait's learned penalty, so the fix has nothing left to fix."
 },
 "game_phase.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "game_phase",
  "title": "Game phase: can the agent switch between farming and hunting?",
  "section": "run output (post-dates the build log)",
  "source_note": "Numbers taken from the experiment run output (tools/model_linq/exp_report.py). Verify against results/model_linq/phase/ before quoting in the report.",
  "question": "Nothing in the feature set says how far through the game we are, so one weight vector averages the crate-farming and kill-hunting regimes instead of switching between them.",
  "preregistration": "experiments/model_linq/PREREG_phase.md",
  "design": {
   "arms": [
    "baseline",
    "+3 phase conjunctions"
   ],
   "n": 40,
   "paired": true,
   "primary_metric": "score",
   "features_added": [
    "x_late_oppdelta",
    "x_late_bomb",
    "x_late_bombopp"
   ],
   "design_note": "the phase enters ONLY as conjunctions - a level is constant across the six actions, so it is absorbed into phi_state and cannot change the argmax"
  },
  "raw_runs": "results/model_linq/phase/",
  "paired_tests": [
   {
    "metric": "score",
    "primary": true,
    "mean": 0.174,
    "t": 1.31
   },
   {
    "metric": "kills",
    "primary": false,
    "mean": 0.027,
    "note": "excludes zero"
   },
   {
    "metric": "invalid",
    "primary": false,
    "mean": 0.505,
    "note": "excludes zero"
   }
  ],
  "verdict": "null on the primary, mechanism confirmed",
  "conclusion": "The mechanism fired - both channels move and exclude zero, and they sum to exactly the +0.174 score change - but the primary does not clear the noise floor. The feature does what it was designed to do and it is not worth points."
 },
 "gamma.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "gamma",
  "title": "Discount factor: does a published gamma transfer?",
  "section": "build log 14",
  "question": "nickstr15 (Heidelberg WS20/21, 23 features) report gamma=0.6 fixing movement loops. Does it help here?",
  "preregistration": "experiments/model_linq/PREREG_gamma.md",
  "design": {
   "arms": [
    "gamma=0.95",
    "gamma=0.80",
    "gamma=0.60"
   ],
   "n": 12,
   "paired": true,
   "primary_metric": "score",
   "primary_contrast": "g60 vs g95",
   "fixed_in_advance": true
  },
  "raw_runs": "results/model_linq/gamma/",
  "arms": [
   {
    "arm": "gamma=0.95",
    "score": 2.71,
    "coins": 1.94,
    "kills": 0.154,
    "steps": 379,
    "invalid": 0.4
   },
   {
    "arm": "gamma=0.80",
    "score": 2.65,
    "coins": 2.02,
    "kills": 0.127,
    "steps": 382,
    "invalid": 0.3
   },
   {
    "arm": "gamma=0.60",
    "score": 0.98,
    "coins": 0.5,
    "kills": 0.098,
    "steps": 375,
    "invalid": 0.1
   }
  ],
  "paired_tests": [
   {
    "metric": "score",
    "primary": true,
    "mean": -1.725,
    "better_in": "0/12",
    "t": -20.18
   },
   {
    "metric": "coins",
    "primary": false,
    "mean": -1.446,
    "better_in": "0/12",
    "t": -19.72
   },
   {
    "metric": "invalid",
    "primary": false,
    "mean": -0.349,
    "better_in": "0/12",
    "t": -15.25
   }
  ],
  "verdict": "strongly negative",
  "conclusion": "The largest effect in the project, and it is a loss. Effective horizon 1/(1-gamma) is 20 steps at 0.95, 5 at 0.80, 2.5 at 0.60, while the features see 10+ steps (d_coin is 0.9^steps). gamma should match how far the features see, so it is a property of the feature set, not a hyperparameter to copy from another paper."
 },
 "herding.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "herding",
  "title": "Herding: can the agent be made to create trap geometry?",
  "section": "build log 14",
  "question": "Kills are capped by opportunity. Can a movement-row feature make the agent create traps?",
  "preregistration": "experiments/model_linq/PREREG_herding.md",
  "design": {
   "arms": [
    "baseline",
    "+opp_confine"
   ],
   "n": 40,
   "paired": true,
   "primary_metric": "kills",
   "one_sided": true,
   "target_effect": 0.04,
   "detectable_at_80pct_power": 0.018,
   "pre_checks": "verified non-redundant and carrying a real gradient before training"
  },
  "raw_runs": "results/model_linq/herd/",
  "paired_tests": [
   {
    "metric": "kills",
    "primary": true,
    "mean": -0.003,
    "sd": 0.041,
    "better_in": "15/40",
    "t": -0.39,
    "ci": [
     -0.016,
     0.01
    ]
   },
   {
    "metric": "coins",
    "primary": false,
    "mean": 0.18,
    "sd": 0.618,
    "better_in": "23/40",
    "t": 1.84,
    "ci": [
     -0.018,
     0.377
    ]
   },
   {
    "metric": "score",
    "primary": false,
    "mean": 0.167,
    "sd": 0.731,
    "better_in": "22/40",
    "t": 1.44,
    "ci": [
     -0.067,
     0.4
    ]
   }
  ],
  "verdict": "bounded null",
  "conclusion": "The one null in the project tight enough to close a question. At n=40 the test could have seen an effect 2.2x smaller than the one worth having; the kills CI is 0.026 wide. \"We found nothing\" and \"there is nothing to find\" are different claims, and only a powered design earns the second."
 },
 "index.json": {
  "project": "Machine Learning Essentials SS2026 - Bomberman",
  "agent": "model_linearQ (Model LinQ)",
  "author": "Sri Vidya Yeluripati",
  "description": "One summary file per experiment: design, pre-registration, aggregate numbers and verdict. Raw per-run results stay in the sibling folders named by each file's \"raw_runs\" field. Generated from the build log; figures are built by tools/model_linq/make_figures.py.",
  "noise_floor": {
   "margin_spread": 1.68,
   "ablation_threshold": 0.38,
   "note": "any claimed effect below these, on a single training run, is not distinguishable from the seed"
  },
  "experiments": [
   {
    "id": "ablations",
    "file": "ablations.json",
    "title": "Feature-group ablations",
    "section": "build log 5",
    "verdict": "two groups load-bearing",
    "n": 3,
    "preregistered": false
   },
   {
    "id": "action_filters",
    "file": "action_filters.json",
    "title": "Action filters from the Pommerman literature",
    "section": "build log 8",
    "verdict": "null",
    "n": null,
    "preregistered": false
   },
   {
    "id": "bc_ceiling",
    "file": "bc_ceiling.json",
    "title": "The ceiling diagnostic: are the features or the learner the constraint?",
    "section": "run output (post-dates the build log)",
    "verdict": "the features are the ceiling",
    "n": null,
    "preregistered": false
   },
   {
    "id": "bombing",
    "file": "bombing.json",
    "title": "Bombing efficiency: five pre-registered arms",
    "section": "build log 11",
    "verdict": "five arms, five nulls on the primary",
    "n": 4,
    "preregistered": true
   },
   {
    "id": "cross_model",
    "file": "cross_model.json",
    "title": "Model LinQ against the team's other models",
    "section": "build log 6",
    "verdict": "Model LinQ is the submitted agent",
    "n": null,
    "preregistered": false
   },
   {
    "id": "deployment_temperature",
    "file": "deployment_temperature.json",
    "title": "Deployment exploration: does a published epsilon transfer?",
    "section": "build log 3 (finding 25)",
    "verdict": "null (greedy retained)",
    "n": null,
    "preregistered": false
   },
   {
    "id": "game_phase",
    "file": "game_phase.json",
    "title": "Game phase: can the agent switch between farming and hunting?",
    "section": "run output (post-dates the build log)",
    "verdict": "null on the primary, mechanism confirmed",
    "n": 40,
    "preregistered": true
   },
   {
    "id": "gamma",
    "file": "gamma.json",
    "title": "Discount factor: does a published gamma transfer?",
    "section": "build log 14",
    "verdict": "strongly negative",
    "n": 12,
    "preregistered": true
   },
   {
    "id": "herding",
    "file": "herding.json",
    "title": "Herding: can the agent be made to create trap geometry?",
    "section": "build log 14",
    "verdict": "bounded null",
    "n": 40,
    "preregistered": true
   },
   {
    "id": "kill_channel",
    "file": "kill_channel.json",
    "title": "opp_trapped: a feature that never worked (exploratory)",
    "section": "build log 13",
    "verdict": "primary null, secondary sent to confirmation",
    "n": 6,
    "preregistered": true
   },
   {
    "id": "kill_channel_confirmatory",
    "file": "kill_channel_confirmatory.json",
    "title": "Confirmatory: the coins effect does not replicate",
    "section": "build log 14",
    "verdict": "null - exploratory effect withdrawn",
    "n": 10,
    "preregistered": true
   },
   {
    "id": "noise_floor",
    "file": "noise_floor.json",
    "title": "Training noise floor",
    "section": "build log 9",
    "verdict": "reference",
    "n": 6,
    "preregistered": false
   },
   {
    "id": "phantom_bombs",
    "file": "phantom_bombs.json",
    "title": "Phantom bombs: cutting the death rate by 44%",
    "section": "build log 14",
    "verdict": "null - no dose-response",
    "n": null,
    "preregistered": false
   },
   {
    "id": "race_gate",
    "file": "race_gate.json",
    "title": "The race gate: bounded null over eleven weight vectors",
    "section": "build log 12",
    "verdict": "null",
    "n": 11,
    "preregistered": false
   },
   {
    "id": "task_ladder",
    "file": "task_ladder.json",
    "title": "The project brief's four-task ladder",
    "section": "build log 14",
    "verdict": "task 2 fails",
    "n": null,
    "preregistered": false
   }
  ]
 },
 "kill_channel.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "kill_channel",
  "title": "opp_trapped: a feature that never worked (exploratory)",
  "section": "build log 13",
  "question": "Does repairing opp_trapped, which has always been an exact copy of opp_in_blast, raise kills?",
  "preregistration": "experiments/model_linq/PREREG_trapped.md",
  "design": {
   "arms": [
    "broken (shipped)",
    "fixed"
   ],
   "n": 6,
   "paired": true,
   "primary_metric": "kills"
  },
  "raw_runs": "results/model_linq/trapped/",
  "defect": {
   "bombs_covering_opponent_broken": 443,
   "claimed_trapping_broken": 443,
   "pct_broken": 100,
   "bombs_covering_opponent_fixed": 502,
   "claimed_trapping_fixed": 23,
   "pct_fixed": 5,
   "note": "the survivability map it queried had opponents removed from the walkable set, so the answer at an opponent tile was always \"cannot escape\""
  },
  "runs": [
   {
    "seed": 3001,
    "kills_base": 0.137,
    "kills_fix": 0.157,
    "score_base": 2.67,
    "score_fix": 3.04,
    "coins_base": 1.99,
    "coins_fix": 2.26
   },
   {
    "seed": 3002,
    "kills_base": 0.237,
    "kills_fix": 0.253,
    "score_base": 3.5,
    "score_fix": 3.62,
    "coins_base": 2.31,
    "coins_fix": 2.35
   },
   {
    "seed": 3003,
    "kills_base": 0.13,
    "kills_fix": 0.297,
    "score_base": 2.15,
    "score_fix": 3.69,
    "coins_base": 1.5,
    "coins_fix": 2.21
   },
   {
    "seed": 3004,
    "kills_base": 0.15,
    "kills_fix": 0.14,
    "score_base": 2.43,
    "score_fix": 2.81,
    "coins_base": 1.68,
    "coins_fix": 2.11
   },
   {
    "seed": 3005,
    "kills_base": 0.13,
    "kills_fix": 0.133,
    "score_base": 2.75,
    "score_fix": 2.74,
    "coins_base": 2.1,
    "coins_fix": 2.07
   },
   {
    "seed": 3006,
    "kills_base": 0.18,
    "kills_fix": 0.167,
    "score_base": 2.57,
    "score_fix": 3.05,
    "coins_base": 1.67,
    "coins_fix": 2.22
   }
  ],
  "paired_tests": [
   {
    "metric": "kills",
    "primary": true,
    "mean": 0.031,
    "better_in": "4/6",
    "t": 1.1
   },
   {
    "metric": "coins",
    "primary": false,
    "mean": 0.328,
    "sd": 0.291,
    "better_in": "5/6",
    "t": 2.76,
    "note": "secondary; promoted to a confirmatory test, which it failed"
   }
  ],
  "verdict": "primary null, secondary sent to confirmation",
  "conclusion": "All four kill weights sat at exactly +0.0427 because the learner cannot split credit between columns it cannot distinguish. Identical weights are a free diagnostic that a deep network does not offer."
 },
 "kill_channel_confirmatory.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "kill_channel_confirmatory",
  "title": "Confirmatory: the coins effect does not replicate",
  "section": "build log 14",
  "question": "Does the +0.328 coins effect from the exploratory run survive a pre-registered test?",
  "preregistration": "experiments/model_linq/PREREG_trapped_confirm.md",
  "design": {
   "arms": [
    "broken (shipped)",
    "fixed"
   ],
   "n": 10,
   "paired": true,
   "primary_metric": "coins",
   "fresh_seeds": true,
   "primary_fixed_in_advance": true
  },
  "raw_runs": "results/model_linq/trapc/",
  "runs": [
   {
    "seed": 4001,
    "kills_base": 0.17,
    "kills_fix": 0.223,
    "score_base": 2.55,
    "score_fix": 3.33,
    "coins_base": 1.7,
    "coins_fix": 2.22
   },
   {
    "seed": 4002,
    "kills_base": 0.11,
    "kills_fix": 0.16,
    "score_base": 2.42,
    "score_fix": 3.24,
    "coins_base": 1.87,
    "coins_fix": 2.44
   },
   {
    "seed": 4003,
    "kills_base": 0.097,
    "kills_fix": 0.177,
    "score_base": 1.08,
    "score_fix": 3.35,
    "coins_base": 0.6,
    "coins_fix": 2.47
   },
   {
    "seed": 4004,
    "kills_base": 0.09,
    "kills_fix": 0.137,
    "score_base": 2.39,
    "score_fix": 2.25,
    "coins_base": 1.94,
    "coins_fix": 1.57
   },
   {
    "seed": 4005,
    "kills_base": 0.153,
    "kills_fix": 0.107,
    "score_base": 2.99,
    "score_fix": 2.11,
    "coins_base": 2.22,
    "coins_fix": 1.58
   },
   {
    "seed": 4006,
    "kills_base": 0.157,
    "kills_fix": 0.083,
    "score_base": 3.0,
    "score_fix": 2.07,
    "coins_base": 2.21,
    "coins_fix": 1.66
   },
   {
    "seed": 4007,
    "kills_base": 0.29,
    "kills_fix": 0.167,
    "score_base": 3.25,
    "score_fix": 3.09,
    "coins_base": 1.8,
    "coins_fix": 2.26
   },
   {
    "seed": 4008,
    "kills_base": 0.13,
    "kills_fix": 0.133,
    "score_base": 2.52,
    "score_fix": 2.61,
    "coins_base": 1.87,
    "coins_fix": 1.95
   },
   {
    "seed": 4009,
    "kills_base": 0.133,
    "kills_fix": 0.13,
    "score_base": 2.64,
    "score_fix": 2.26,
    "coins_base": 1.97,
    "coins_fix": 1.61
   },
   {
    "seed": 4010,
    "kills_base": 0.32,
    "kills_fix": 0.133,
    "score_base": 3.7,
    "score_fix": 2.79,
    "coins_base": 2.1,
    "coins_fix": 2.12
   }
  ],
  "paired_tests": [
   {
    "metric": "coins",
    "primary": true,
    "mean": 0.16,
    "sd": 0.748,
    "better_in": "6/10",
    "t": 0.68,
    "ci": [
     -0.375,
     0.695
    ]
   },
   {
    "metric": "kills",
    "primary": false,
    "mean": -0.02,
    "sd": 0.087,
    "better_in": "5/10",
    "t": -0.73,
    "ci": [
     -0.082,
     0.042
    ]
   }
  ],
  "verdict": "null - exploratory effect withdrawn",
  "power_note": {
   "prereg_promised_power": 0.95,
   "prereg_assumed_sd": 0.291,
   "true_sd": 0.748,
   "true_power": 0.36,
   "lesson": "never power a confirmatory study on the variance estimate from the exploratory run that motivated it - the winner's curse inflates the effect and deflates the spread at the same time"
  },
  "conclusion": "The effect halved and the sd was wrong by 2.6x. Sixteen trainings across two pre-registered experiments retired a plausible finding before it reached the report."
 },
 "noise_floor.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "noise_floor",
  "title": "Training noise floor",
  "section": "build log 9",
  "question": "How large is an effect allowed to be before it counts as real?",
  "design": {
   "arms": [
    "identical recipe"
   ],
   "n": 6,
   "paired": false,
   "note": "six independent trainings, identical recipe and hyperparameters, different seeds"
  },
  "raw_runs": "results/model_linq/noise/",
  "runs": [
   {
    "seed": 1,
    "margin": -0.51,
    "score": 2.45,
    "coins": 1.25,
    "crates": 22.3,
    "bombs": 41.1,
    "crates_per_bomb": 0.54
   },
   {
    "seed": 2,
    "margin": 0.35,
    "score": 3.01,
    "coins": 2.25,
    "crates": 24.1,
    "bombs": 47.3,
    "crates_per_bomb": 0.51
   },
   {
    "seed": 3,
    "margin": 0.33,
    "score": 2.99,
    "coins": 2.29,
    "crates": 23.7,
    "bombs": 50.4,
    "crates_per_bomb": 0.47
   },
   {
    "seed": 4,
    "margin": -0.62,
    "score": 2.31,
    "coins": 1.49,
    "crates": 15.5,
    "bombs": 51.1,
    "crates_per_bomb": 0.3
   },
   {
    "seed": 5,
    "margin": 1.06,
    "score": 3.62,
    "coins": 2.52,
    "crates": 30.4,
    "bombs": 33.9,
    "crates_per_bomb": 0.9
   },
   {
    "seed": 6,
    "margin": -0.06,
    "score": 2.71,
    "coins": 2.06,
    "crates": 22.2,
    "bombs": 50.6,
    "crates_per_bomb": 0.44
   }
  ],
  "summary": {
   "margin_mean": 0.09,
   "margin_sd": 0.63,
   "margin_spread": 1.68,
   "score_mean": 2.85,
   "coins_mean": 1.98,
   "crates_mean": 23.0,
   "bombs_mean": 45.7
  },
  "verdict": "reference",
  "conclusion": "Training variance alone spans 1.68 margin. Any claimed effect smaller than that, measured on a single training run, is not distinguishable from the seed."
 },
 "phantom_bombs.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "phantom_bombs",
  "title": "Phantom bombs: cutting the death rate by 44%",
  "section": "build log 14",
  "question": "Suicides are 0.010/round against our own agents and 0.372 with one rule_based_agent, so the killing bombs are somebody else's. Does anticipating them help?",
  "design": {
   "arms": [
    "base",
    "K=2",
    "K=3",
    "K=5",
    "bomb filter",
    "both"
   ],
   "evaluation": "250 rounds, mixed tournament field",
   "evaluation_noise": 0.15
  },
  "arms": [
   {
    "arm": "base",
    "score": 4.71,
    "vs_base": 0.0,
    "coins": 3.79,
    "kills": 0.184,
    "suicides": 0.372,
    "suicide_cut_pct": 0
   },
   {
    "arm": "K=2",
    "score": 4.88,
    "vs_base": 0.17,
    "coins": 3.7,
    "kills": 0.236,
    "suicides": 0.352,
    "suicide_cut_pct": 5
   },
   {
    "arm": "K=3",
    "score": 4.74,
    "vs_base": 0.03,
    "coins": 3.72,
    "kills": 0.204,
    "suicides": 0.248,
    "suicide_cut_pct": 33
   },
   {
    "arm": "K=5",
    "score": 4.74,
    "vs_base": 0.03,
    "coins": 3.64,
    "kills": 0.22,
    "suicides": 0.208,
    "suicide_cut_pct": 44
   },
   {
    "arm": "bomb filter",
    "score": 4.34,
    "vs_base": -0.37,
    "coins": 3.48,
    "kills": 0.172,
    "suicides": 0.188,
    "suicide_cut_pct": null
   },
   {
    "arm": "both",
    "score": 4.39,
    "vs_base": -0.32,
    "coins": 3.61,
    "kills": 0.156,
    "suicides": 0.132,
    "suicide_cut_pct": 64
   }
  ],
  "verdict": "null - no dose-response",
  "conclusion": "Suicides fall monotonically with K and score does not follow. The best-scoring arm (K=2) cuts suicides the least, so +0.17 is a draw from the noise. A suicide also costs no points directly: score is coins + 5*kills, and the agent dies at step ~344 of 400 having already collected most of what it will collect."
 },
 "race_gate.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "race_gate",
  "title": "The race gate: bounded null over eleven weight vectors",
  "section": "build log 12",
  "question": "Should the agent decline bombs whose released coins an opponent would reach first?",
  "design": {
   "arms": [
    "gate off",
    "gate on"
   ],
   "n": 11,
   "paired": true,
   "primary_metric": "margin",
   "note": "eleven independently trained weight vectors, because a play-time method claim needs several weight draws, not one"
  },
  "raw_runs": "results/model_linq/race/",
  "paired_tests": [
   {
    "metric": "bombs per round",
    "mean": -13.1,
    "better_in": "0/11",
    "t": -10.59
   },
   {
    "metric": "crates per bomb",
    "mean": 0.23,
    "better_in": "11/11",
    "t": 3.32
   },
   {
    "metric": "coins",
    "mean": -0.02,
    "better_in": "5/11",
    "t": -0.63
   },
   {
    "metric": "margin",
    "primary": true,
    "mean": 0.02,
    "better_in": "5/11",
    "t": 0.27
   },
   {
    "metric": "score",
    "mean": 0.0,
    "better_in": "5/11",
    "t": 0.07
   }
  ],
  "verdict": "null",
  "conclusion": "The gate does exactly what it was designed to do - bombs fall by 13 per round, crates per bomb rise in 11 of 11 - and the score does not move at all."
 },
 "task_ladder.json": {
  "agent": "model_linearQ",
  "model": "Model LinQ (linear Q, Expected SARSA(lambda))",
  "id": "task_ladder",
  "title": "The project brief's four-task ladder",
  "section": "build log 14",
  "question": "The brief defines four staged tasks. The agent had only ever been measured on the last one.",
  "design": {
   "evaluation": "shipped weights, learning off, one protocol across all tasks"
  },
  "tasks": [
   {
    "task": "1  coin-heaven, solo",
    "coins": 50.0,
    "of": 50,
    "kills": 0.0,
    "suicides": 0.0,
    "score": 50.0,
    "best_opponent": null,
    "steps": 125
   },
   {
    "task": "2  classic, solo (crates)",
    "coins": 5.08,
    "of": 9,
    "kills": 0.0,
    "suicides": 0.0,
    "score": 5.08,
    "best_opponent": null,
    "steps": 400
   },
   {
    "task": "3a classic vs 3x peaceful",
    "coins": 4.73,
    "of": 9,
    "kills": 0.9,
    "suicides": 0.02,
    "score": 9.23,
    "best_opponent": 0.03,
    "steps": 396
   },
   {
    "task": "3b classic vs 3x coin_collector",
    "coins": 2.07,
    "of": 9,
    "kills": 0.12,
    "suicides": 0.053,
    "score": 2.67,
    "best_opponent": 2.73,
    "steps": 378
   },
   {
    "task": "4  classic vs 3x rule_based",
    "coins": 2.58,
    "of": 9,
    "kills": 0.13,
    "suicides": 0.447,
    "score": 3.23,
    "best_opponent": 2.73,
    "steps": 304
   }
  ],
  "verdict": "task 2 fails",
  "conclusion": "Task 1 is perfect. Task 2 is not: solo, unopposed, in no danger, with the whole clock, the agent finds 5.08 of 9 coins and uses all 400 steps. This is a solo shortfall, so none of the multi-agent explanations apply. It went unnoticed all project because a four-player game hides it. Suicides are 0.020 vs peaceful and 0.447 vs rule_based, which localises the deaths to interaction with a bombing opponent, not to our own danger model."
 }
}
""")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, obj in SUMMARIES.items():
        with open(os.path.join(OUT, name), 'w') as f:
            json.dump(obj, f, indent=2)
        print('wrote', os.path.join(OUT, name))
    print(f'\n{len(SUMMARIES)} files in {OUT}')


if __name__ == '__main__':
    main()
