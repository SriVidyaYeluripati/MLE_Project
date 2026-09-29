#Build the cross-model comparison table from the yardstick result files_needed.
import json, os

FILES_needed = [
    ("LinearQ", "results/cross_model/yardstick_linearQ.json"),
    ("ForestQ", "results/cross_model/yardstick_model_a.json"),
    ("GBT",     "results/cross_model/yardstick_gbt.json"),
    ("DQN",     "results/cross_model/yardstick_dqn.json"),
        ]
rows = []

for label, path in FILES_needed:
    if not os.path.isfile(path):
        print(f"missing: {path}")
        continue
    by = json.load(open(path))["by_agent"]
    opp = {k: v for k, v in by.items() if k.startswith("rule_based")}
    own = {k: v for k, v in by.items() if not k.startswith("rule_based")}
    if len(own) != 1 or not opp:
        print(f"skipped {path}: expected 1 agent + rule_based opponents, got {list(by)}")
        continue
    name, a = next(iter(own.items()))
    n = a.get("rounds", 1)
    ours = a.get("score", 0) / n
    opp_rates = sorted((v.get("score", 0) / v.get("rounds", n) for v in opp.values()), reverse=True)
    rows.append(dict(
        label=label, agent=name, n=n, score=ours, margin=ours - opp_rates[0],
        coins=a.get("coins", 0) / n, kills=a.get("kills", 0) / n,
        suic=a.get("suicides", 0) / n, inval=a.get("invalid", 0) / n,
        spread=(max(opp_rates) - min(opp_rates)) if len(opp_rates) > 1 else float("nan"),
    ))

rows.sort(key=lambda r: -r["margin"])

print(f"\n{'model':<10}{'rounds':>7}{'score/rnd':>11}{'margin':>9}{'coins':>8}{'kills':>8}{'suic':>7}{'inval':>8}")
for r in rows:
    print(f"{r['label']:<10}{r['n']:>7}{r['score']:>11.2f}{r['margin']:>+9.2f}"
          f"{r['coins']:>8.2f}{r['kills']:>8.3f}{r['suic']:>7.3f}{r['inval']:>8.2f}")

sp = [r["spread"] for r in rows if r["spread"] == r["spread"]]
if sp:
    print(f"\nEvaluation noise: the rule-based opponents within one run differ by "
          f"{min(sp):.2f}-{max(sp):.2f} points per round.")