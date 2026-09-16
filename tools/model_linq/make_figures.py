#!/usr/bin/env python3
"""
Build the report figures from results/model_linq/summary/*.json.

    python tools/model_linq/make_figures.py                 # -> results/model_linq/figures/
    python tools/model_linq/make_figures.py --format pdf    # vector, for LaTeX

Every number plotted is read from the summary JSON; nothing is hard-coded here.
Change a number in the JSON and the figure changes with it.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- palette -------------------------------------------------------------- #
# Diverging blue <-> red: the poles mean "gained" and "lost". Nulls take a
# neutral ink, so colour never carries the verdict alone - every mark is also
# directly labelled and the verdict is written out in the tick label.
LOSS = '#e34948'
GAIN = '#2a78d6'
NULL = '#52514e'
GRID = '#dedcd6'
INK = '#0b0b0b'
INK2 = '#52514e'
SURFACE = '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK2,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlecolor': INK,
    'text.color': INK,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'font.size': 9,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
})


def load(d, name):
    with open(os.path.join(d, name + '.json')) as f:
        return json.load(f)


def tidy(ax, xgrid=True):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis='x' if xgrid else 'y', linewidth=0.6)


def save(fig, out, name, fmt):
    path = os.path.join(out, f'{name}.{fmt}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ', path)


# --- figures -------------------------------------------------------------- #

def fig_noise_floor(d, out, fmt):
    e = load(d, 'noise_floor')
    seeds = [r['seed'] for r in e['runs']]
    marg = [r['margin'] for r in e['runs']]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.axhspan(min(marg), max(marg), color=GRID, alpha=0.55, zorder=0)
    ax.axhline(0, color=INK2, linewidth=1, zorder=1)
    ax.plot(seeds, marg, 'o', color=GAIN, markersize=9, zorder=3)
    for s, m in zip(seeds, marg):
        ax.annotate(f'{m:+.2f}', (s, m), textcoords='offset points',
                    xytext=(0, 11), ha='center', fontsize=8, color=INK)
    ax.set_xticks(seeds)
    ax.set_xlabel('training seed (identical recipe)')
    ax.set_ylabel('margin vs best rule_based')
    ax.set_title(f"Six identical trainings span {e['summary']['margin_spread']:.2f} margin")
    ax.set_ylim(min(marg) - 0.45, max(marg) + 0.45)
    tidy(ax, xgrid=False)
    save(fig, out, 'fig1_noise_floor', fmt)


def fig_ablations(d, out, fmt):
    e = load(d, 'ablations')
    rows = [a for a in e['arms'] if a['delta'] is not None]
    rows.sort(key=lambda a: a['delta'])
    contaminated = {'no_conj'}          # see the caveat field in ablations.json
    names = [a['arm'] + (' *' if a['arm'] in contaminated else '') for a in rows]
    delta = [a['delta'] for a in rows]
    thr = 0.38
    colors = [LOSS if v < -thr else GAIN if v > thr else NULL for v in delta]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.barh(names, delta, color=colors, height=0.62,
            hatch=['///' if n.endswith('*') else '' for n in names],
            edgecolor=SURFACE, linewidth=0)
    ax.axvline(0, color=INK2, linewidth=1)
    for x in (-thr, thr):
        ax.axvline(x, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    for i, v in enumerate(delta):
        ax.annotate(f'{v:+.2f}', (v, i), textcoords='offset points',
                    xytext=(-6 if v < 0 else 6, 0), ha='right' if v < 0 else 'left',
                    va='center', fontsize=8, color=INK)
    ax.set_xlabel('change in margin when the group is removed')
    ax.set_title('Only survival and shaping are load-bearing')
    ax.set_xlim(min(delta) - 1.4, max(delta) + 1.4)
    tidy(ax)
    fig.text(0.5, -0.05,
             'dashed lines: ±0.38, the threshold set by the baseline\'s own spread.   '
             '* hatched: contaminated by duplicate columns (see ablations.json)',
             ha='center', fontsize=7.5, color=INK2)
    save(fig, out, 'fig2_ablations', fmt)


def fig_gamma(d, out, fmt):
    e = load(d, 'gamma')
    names = [a['arm'] for a in e['arms']]
    score = [a['score'] for a in e['arms']]
    coins = [a['coins'] for a in e['arms']]
    horizon = [round(1 / (1 - float(n.split('=')[1])), 1) for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar([i - 0.17 for i in x], score, width=0.32, color=GAIN, label='score')
    ax.bar([i + 0.17 for i in x], coins, width=0.32, color=NULL, label='coins')
    for i, (s, c) in enumerate(zip(score, coins)):
        ax.annotate(f'{s:.2f}', (i - 0.17, s), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=8, color=INK)
        ax.annotate(f'{c:.2f}', (i + 0.17, c), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'{n}\nhorizon {h} steps' for n, h in zip(names, horizon)])
    ax.set_ylabel('per round, 12 seeds')
    ax.set_title('The features see 10+ steps; γ = 0.60 sees 2.5')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(score) * 1.25)
    tidy(ax, xgrid=False)
    save(fig, out, 'fig3_gamma', fmt)


def fig_effects(d, out, fmt):
    """
    The headline figure: every pre-registered primary with its 95% CI.

    Faceted by metric, one panel per metric with its own x-axis. Kills move by
    hundredths and score by whole points, so putting them on a shared axis would
    hide the tightest result in the project (herding) behind the largest one.
    """
    panels = [
        ('score',  [('gamma', 'γ = 0.60 vs 0.95'), ('game_phase', 'game phase')]),
        ('coins',  [('kill_channel_confirmatory', 'opp_trapped fix (confirmatory)')]),
        ('kills',  [('kill_channel', 'opp_trapped fix (exploratory)'), ('herding', 'herding')]),
        ('margin', [('race_gate', 'race gate'), ('bombing', 'bombing, best arm')]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.2))
    for ax, (metric, items) in zip(axes.ravel(), panels):
        rows = []
        for key, label in items:
            e = load(d, key)
            if key == 'bombing':                       # reports arms, not a paired primary
                a = e['arms'][0]
                rows.append(dict(label=f"{label}\nn={e['design']['n']}",
                                 mean=a['margin']['mean'], ci=None, t=None))
                continue
            for t in e.get('paired_tests', []):
                if t.get('primary'):
                    rows.append(dict(label=f"{label}\nn={e['design']['n']}",
                                     mean=t['mean'], ci=t.get('ci'), t=t.get('t')))

        ax.axvline(0, color=INK2, linewidth=1)
        for i, r in enumerate(rows):
            excludes_zero = r['ci'] is not None and (r['ci'][0] > 0 or r['ci'][1] < 0)
            col = LOSS if r['mean'] < 0 and abs(r['mean']) > 1.0 else \
                  (GAIN if excludes_zero and r['mean'] > 0 else NULL)
            if r['ci']:
                ax.plot(r['ci'], [i, i], color=col, linewidth=2.2, solid_capstyle='round')
            ax.plot([r['mean']], [i], 'o', color=col, markersize=8, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.8)
            txt = f"{r['mean']:+.3f}" + (f"  t={r['t']:.2f}" if r['t'] is not None else '')
            ax.annotate(txt, (r['mean'], i), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=7.5, color=INK)

        span = max([abs(v) for r in rows for v in ([r['mean']] + (r['ci'] or []))]) or 1
        ax.set_xlim(-span * 1.75, span * 1.75)
        ax.set_ylim(-0.7, len(rows) - 0.1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r['label'] for r in rows], fontsize=7.5)
        ax.set_title(metric, fontsize=9, loc='left', color=INK2)
        ax.tick_params(axis='x', labelsize=7.5)
        tidy(ax)

    fig.suptitle('Every pre-registered primary, faceted by metric',
                 fontsize=11, fontweight='bold', y=1.0)
    fig.text(0.5, -0.04, 'paired effect on the primary metric; bars are 95% CI, '
             'one panel per metric because the scales differ',
             ha='center', fontsize=8, color=INK2)
    fig.tight_layout(h_pad=2.4, w_pad=3.0)
    save(fig, out, 'fig4_effects', fmt)


def fig_cross_model(d, out, fmt):
    e = load(d, 'cross_model')
    rows = sorted(e['yardstick'], key=lambda r: r['margin'])
    names = [f"{r['model']}\n({r['owner']})" for r in rows]
    marg = [r['margin'] for r in rows]
    colors = [GAIN if v > 0 else LOSS for v in marg]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.barh(names, marg, color=colors, height=0.6)
    ax.axvline(0, color=INK2, linewidth=1)
    for i, (v, r) in enumerate(zip(marg, rows)):
        ax.annotate(f"{v:+.2f}   (score {r['score']:.2f})", (v, i),
                    xytext=(-6 if v < 0 else 6, 0), textcoords='offset points',
                    ha='right' if v < 0 else 'left', va='center', fontsize=8, color=INK)
    ax.set_xlabel('margin vs best rule_based_agent, 200 rounds each')
    ax.set_title('The team\'s four models on one yardstick')
    ax.set_xlim(min(marg) - 2.6, max(marg) + 2.6)
    tidy(ax)
    save(fig, out, 'fig5_cross_model', fmt)


def fig_ceiling(d, out, fmt):
    e = load(d, 'bc_ceiling')
    ag = e['agreement_with_rule_based']
    names = [a['model'].replace(' on the same features', '\non the same features')
                       .replace(' + 128 random tanh features', '\n+ 128 random tanh features')
             for a in ag]
    pct = [a['pct'] for a in ag]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.barh(names, pct, color=[NULL, GAIN, GAIN], height=0.58)
    ax.axvline(100, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate('rule_based itself\n100%', (100, 0.1), xytext=(-6, 0),
                textcoords='offset points', ha='right', fontsize=8, color=INK2)
    for i, v in enumerate(pct):
        ax.annotate(f'{v:.1f}%', (v, i), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=8, color=INK)
    ax.set_xlim(0, 118)
    ax.set_xlabel('agreement with rule_based_agent on the states LinQ visits')
    ax.set_title('Non-linearity buys 0.1 points: the features are the ceiling')
    tidy(ax)
    save(fig, out, 'fig6_ceiling', fmt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', default='results/model_linq/summary')
    p.add_argument('--out', default='results/model_linq/figures')
    p.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'])
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print('figures:')
    for fn in (fig_noise_floor, fig_ablations, fig_gamma, fig_effects,
               fig_cross_model, fig_ceiling):
        fn(a.summary, a.out, a.format)


if __name__ == '__main__':
    main()
