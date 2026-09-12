import json,os,sys,numpy as np
from math import sqrt
def load(t):
    f=f'results/model_linq/trapc/{t}.json'
    if not os.path.exists(f): return None
    d=json.load(open(f))['by_agent']; me=d['model_linearQ']; n=me['rounds']
    g=lambda k: me.get(k,0)/n
    best=max(v['score']/v['rounds'] for k,v in d.items() if k!='model_linearQ')
    return dict(kills=g('kills'),score=g('score'),coins=g('coins'),
                bombs=g('bombs'),margin=g('score')-best)
S=list(range(1,11))
b=[load(f'cbase{s}') for s in S]; f=[load(f'cfix{s}') for s in S]
ok=[i for i in range(len(S)) if b[i] and f[i]]
if not ok: print('no completed seed pairs yet'); raise SystemExit
print(f"{'seed':5s} {'kills base':>10s} {'fix':>6s} | {'score base':>10s} {'fix':>6s} | {'coins':>6s} {'fix':>6s}")
for i in ok:
    print(f"{S[i]:<5d} {b[i]['kills']:10.3f} {f[i]['kills']:6.3f} | "
          f"{b[i]['score']:10.2f} {f[i]['score']:6.2f} | {b[i]['coins']:6.2f} {f[i]['coins']:6.2f}")
n=len(ok)
from scipy import stats as _st
crit=_st.t.ppf(.95,n-1)
print(f"\npaired, n={n}  PRE-REGISTERED: coins is primary, one-sided, t > {crit:.3f} at df={n-1}")
if n<10: print(f"  NOTE: n={n} of the pre-registered 10. Report as an incomplete run with power recomputed.")
for k in ('kills','score','coins','bombs','margin'):
    d=np.array([f[i][k]-b[i][k] for i in ok])
    t=d.mean()/(d.std(ddof=1)/sqrt(n)) if d.std(ddof=1)>0 else float('nan')
    print(f"  {k:7s} mean {d.mean():+7.3f}  sd {d.std(ddof=1):6.3f}  better in {int((d>0).sum())}/{n}  t={t:+5.2f}"
          f"  {'<== PASSES' if k=='coins' and t>crit else ('<== FAILS' if k=='coins' else '')}")
