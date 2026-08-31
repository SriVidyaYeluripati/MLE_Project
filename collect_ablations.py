"""Aggregate results/ablation/<variant>__r<rep>.json into the report table."""
import glob, json, os, statistics as st

ORDER = ['full', 'no_escape', 'no_opp', 'no_conj', 'no_shaping', 'no_mask',
         'no_danger', 'no_crates', 'no_coins', 'no_trapped', 'no_geometry',
         'no_phi_state', 'no_traces']
FIELDS = ['score', 'margin', 'coins', 'kills', 'crates', 'suic', 'inval', 'alive']

def one(path):
    d = json.load(open(path))['by_agent']
    me = next(k for k in d if 'linearQ' in k)
    r = d[me]; n = r['rounds']
    rb = max(v['score'] / v['rounds'] for k, v in d.items() if 'rule_based' in k)
    return {'score': r['score']/n, 'margin': r['score']/n - rb,
            'coins': r['coins']/n, 'kills': r['kills']/n, 'crates': r['crates']/n,
            'suic': r['suicides']/n, 'inval': r['invalid']/n, 'alive': r['steps']/n}

runs = {}
for p in sorted(glob.glob('results/ablation/*.json')):
    runs.setdefault(os.path.basename(p)[:-5].split('__r')[0], []).append(one(p))
if not runs:
    raise SystemExit('no results yet')

agg = {}
for name, rs in runs.items():
    a = {'n': len(rs)}
    for f in FIELDS:
        v = [r[f] for r in rs]
        a[f] = st.mean(v)
        a[f+'_sd'] = st.stdev(v) if len(v) > 1 else float('nan')
        a[f+'_lo'], a[f+'_hi'] = min(v), max(v)
    agg[name] = a

names = [n for n in ORDER if n in agg] + [n for n in agg if n not in ORDER]
base = agg.get('full')
head = (f"{'variant':13s}{'n':>3s}{'margin':>8s}{'range':>15s}{'delta':>8s}"
        f"{'score':>7s}{'coins':>7s}{'kills':>7s}{'crates':>8s}"
        f"{'suic':>7s}{'inval':>8s}{'alive':>8s}")
print(head); print('-' * len(head))
for n in names:
    a = agg[n]
    rng = f"{a['margin_lo']:+.2f}..{a['margin_hi']:+.2f}" if a['n'] > 1 else ''
    delta = '' if base is None or n == 'full' else f"{a['margin']-base['margin']:+8.2f}"
    print(f"{n:13s}{a['n']:3d}{a['margin']:8.2f}{rng:>15s}{delta:>8s}"
          f"{a['score']:7.2f}{a['coins']:7.2f}{a['kills']:7.2f}{a['crates']:8.2f}"
          f"{a['suic']:7.2f}{a['inval']:8.2f}{a['alive']:8.1f}")
if base and base['n'] > 1:
    sd = base['margin_sd']
    print(f"\nbaseline margin {base['margin']:+.2f} +/- {sd:.2f} over {base['n']} runs "
          f"(range {base['margin_lo']:+.2f}..{base['margin_hi']:+.2f})")
    print(f"claim a delta only past roughly +/-{2*sd:.2f}.")
else:
    print("\nrun the baseline more than once - without its spread no delta is readable.")
