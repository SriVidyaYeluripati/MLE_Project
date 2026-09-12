import json, os, glob
def row(path, me='model_linearQ'):
    d = json.load(open(path))['by_agent']
    best = max(v['score']/v['rounds'] for k, v in d.items() if k != me)
    r = d[me]; n = r['rounds']; g = lambda k: r.get(k, 0)/n
    return dict(margin=g('score')-best, score=g('score'), kills=g('kills'),
                coins=g('coins'), crates=g('crates'), suic=g('suicides'), n=n)
for opp in ('rb','cc'):
    print(f"\n=== vs 3x {'rule_based' if opp=='rb' else 'coin_collector'} ===")
    for tag in ('prepool','pool'):
        for f in sorted(glob.glob(f'results/pool2/{tag}_{opp}_*.json')):
            r = row(f); rep = f.split('_')[-1].split('.')[0]
            print(f"  {tag:8s} rep{rep}  margin {r['margin']:+6.2f}  score {r['score']:5.2f}  "
                  f"kills {r['kills']:4.2f}  coins {r['coins']:5.2f}  crates {r['crates']:6.2f}  "
                  f"suic {r['suic']:4.2f}  ({r['n']} rds)")
