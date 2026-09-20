import json, os, numpy as np
ARMS = [('base','control'), ('drop','drop-time crate reward'),
        ('tgt','selective crate targets'), ('both','drop + targets'),
        ('filt','trained inside the bomb filter')]
SEEDS = [1,2,3,4]
def load(tag):
    f=f'results/bomb/{tag}.json'
    if not os.path.exists(f): return None
    d=json.load(open(f))['by_agent']; me=d['model_linearQ']; n=me['rounds']
    g=lambda k: me.get(k,0)/n
    best=max(v['score']/v['rounds'] for k,v in d.items() if k!='model_linearQ')
    return dict(coins=g('coins'), crates=g('crates'), bombs=g('bombs'),
                crpb=g('crates')/max(g('bombs'),1e-9), suic=g('suicides'),
                margin=g('score')-best, alive=g('steps'))
print(f"{'arm':32s} {'cr/bomb':>16s} {'coins':>15s} {'bombs':>7s} {'suic':>5s} {'margin':>16s}")
res={}
for tag,lbl in ARMS:
    rows=[load(f'{tag}{s}') for s in SEEDS]
    rows=[r for r in rows if r]
    if not rows: print(f"{lbl:32s}  (none yet)"); continue
    m=lambda k: np.array([r[k] for r in rows])
    res[tag]={k:m(k) for k in rows[0]}
    f=lambda k: f"{m(k).mean():6.2f}+-{m(k).std(ddof=1) if len(rows)>1 else 0:.2f}"
    print(f"{lbl:32s} {f('crpb'):>16s} {f('coins'):>15s} {m('bombs').mean():7.1f} "
          f"{m('suic').mean():5.2f} {f('margin'):>16s}   n={len(rows)}")
if 'base' in res and len(res['base']['crpb'])>1:
    print("\n--- decision rule (fixed in PREREG before the runs) ---")
    print("an arm improves only if its mean beats base by more than the larger sd,")
    print("on crates-per-bomb AND without losing coins\n")
    b=res['base']
    for tag,lbl in ARMS[1:]:
        if tag not in res or len(res[tag]['crpb'])<2: continue
        a=res[tag]
        d_c=a['crpb'].mean()-b['crpb'].mean(); sd=max(a['crpb'].std(ddof=1),b['crpb'].std(ddof=1))
        d_k=a['coins'].mean()-b['coins'].mean()
        ok = d_c>sd and d_k>=-0.05
        print(f"  {lbl:32s} d(cr/bomb) {d_c:+6.2f} vs sd {sd:.2f} | d(coins) {d_k:+5.2f}"
              f"  -> {'IMPROVEMENT' if ok else 'no effect'}")
