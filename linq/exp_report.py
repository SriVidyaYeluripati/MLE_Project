"""Paired analysis for the two pre-registered experiments. Reads only the
metric each pre-registration named as primary, and says PASS or FAIL by the
rule fixed in advance."""
import json, os, sys, numpy as np
from scipy import stats

def load(exp, tag, s):
    f = f'linq/results/{exp}/{tag}{s}.json'
    if not os.path.exists(f): return None
    d = json.load(open(f))['by_agent']; me = d['model_linearQ']; n = me['rounds']
    if not n: return None
    g = lambda k: me.get(k, 0) / n
    best = max(v['score']/v['rounds'] for k, v in d.items() if k != 'model_linearQ')
    return dict(kills=g('kills'), score=g('score'), coins=g('coins'),
                suicides=g('suicides'), steps=g('steps'), invalid=g('invalid'),
                margin=g('score') - best)

def paired(a, b, keys, seeds):
    ok = [i for i in range(len(seeds)) if a[i] and b[i]]
    return ok, {k: np.array([b[i][k] - a[i][k] for i in ok]) for k in keys}

def line(k, d, crit=None, one=True, primary=False):
    n = len(d); sd = d.std(ddof=1)
    t = d.mean()/(sd/np.sqrt(n)) if sd > 0 else float('nan')
    se = sd/np.sqrt(n); c = stats.t.ppf(.975, n-1)
    mark = ''
    if primary:
        ok = (t > crit) if one else (abs(t) > crit)
        mark = '   <== PASSES' if ok else '   <== FAILS'
    print(f"  {k:9s} mean {d.mean():+7.3f}  sd {sd:6.3f}  better {int((d>0).sum()):>2d}/{n}"
          f"  t={t:+5.2f}  95% CI [{d.mean()-c*se:+.3f}, {d.mean()+c*se:+.3f}]{mark}")

KEYS = ('kills','score','coins','suicides','steps','invalid','margin')

if sys.argv[1] == 'gamma':
    S = list(range(1, 13))
    arms = {t: [load('gamma', t, s) for s in S] for t in ('g95','g80','g60')}
    done = {t: sum(x is not None for x in v) for t, v in arms.items()}
    print(f"\n=== GAMMA ===  complete seeds: " +
          "  ".join(f"{t} {done[t]}/12" for t in ('g95','g80','g60')))
    for t in ('g95','g80','g60'):
        v = [x for x in arms[t] if x]
        if v: print(f"  {t}: score {np.mean([x['score'] for x in v]):.2f}  "
                    f"coins {np.mean([x['coins'] for x in v]):.2f}  "
                    f"kills {np.mean([x['kills'] for x in v]):.3f}  "
                    f"steps {np.mean([x['steps'] for x in v]):.0f}  "
                    f"invalid {np.mean([x['invalid'] for x in v]):.1f}")
    ok, d = paired(arms['g95'], arms['g60'], KEYS, S)
    if len(ok) < 3: print("\nnot enough paired seeds yet"); sys.exit()
    crit = stats.t.ppf(.975, len(ok)-1)
    print(f"\nPRIMARY: g60 minus g95, score, TWO-sided, |t| > {crit:.3f} at df={len(ok)-1}")
    if len(ok) < 12: print(f"  NOTE: {len(ok)} of the pre-registered 12.")
    for k in KEYS: line(k, d[k], crit, one=False, primary=(k == 'score'))
    ok2, d2 = paired(arms['g95'], arms['g80'], ('score',), S)
    if ok2: print(f"\n  (g80 minus g95, score {d2['score'].mean():+.3f} "
                  f"-- intermediate point, NOT a test)")

elif sys.argv[1] == 'phase':
    S = list(range(1, 41))
    a = [load('phase','pbase',s) for s in S]; b = [load('phase','pphase',s) for s in S]
    ok, d = paired(a, b, KEYS, S)
    print(f"\n=== GAME PHASE ===  complete pairs: {len(ok)}/40   (vs 3x rule_based_agent)")
    if len(ok) < 3: print("not enough paired seeds yet"); sys.exit()
    crit = stats.t.ppf(.95, len(ok)-1)
    print(f"PRIMARY: score, ONE-sided, t > {crit:.3f} at df={len(ok)-1}")
    if len(ok) < 40:
        print(f"  NOTE: {len(ok)} of the pre-registered 40 -- incomplete, "
              f"detects {2.8*d['score'].std(ddof=1)/np.sqrt(len(ok)):+.3f} score at 80% power")
    for k in KEYS: line(k, d[k], crit, one=True, primary=(k == 'score'))

elif sys.argv[1] == 'herd':
    S = list(range(1, 41))
    a = [load('herd','hbase',s) for s in S]; b = [load('herd','hherd',s) for s in S]
    ok, d = paired(a, b, KEYS, S)
    print(f"\n=== HERDING ===  complete pairs: {len(ok)}/40")
    if len(ok) < 3: print("not enough paired seeds yet"); sys.exit()
    crit = stats.t.ppf(.95, len(ok)-1)
    print(f"PRIMARY: kills, ONE-sided, t > {crit:.3f} at df={len(ok)-1}")
    if len(ok) < 40:
        print(f"  NOTE: {len(ok)} of the pre-registered 40 -- incomplete, "
              f"detects {2.8*d['kills'].std(ddof=1)/np.sqrt(len(ok)):+.3f} kills at 80% power")
    for k in KEYS: line(k, d[k], crit, one=True, primary=(k == 'kills'))
