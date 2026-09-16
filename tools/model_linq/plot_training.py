#!/usr/bin/env python3
"""
Plot training progress from the CSV that train.py writes.

    python tools/model_linq/plot_training.py results/model_linq/training/run1.csv
    python tools/model_linq/plot_training.py run1.csv --smooth 15 --format pdf

Produces two figures next to the CSV:

  <name>_progress   score and survival, stacked, sharing one x-axis
  <name>_panels     six metrics, one panel each, raw plus a smoothed line
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SERIES = '#2a78d6'      # the smoothed trend
RAW = '#a9a79e'         # the unsmoothed series behind it
GRID = '#dedcd6'
INK = '#0b0b0b'
INK2 = '#52514e'
SURFACE = '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
    'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'font.size': 9, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'legend.fontsize': 8,
})

PANELS = [
    ('score', 'score per round'),
    ('coins', 'coins per round'),
    ('crates', 'crates per round'),
    ('suicides', 'suicides per round'),
    ('survived', 'rounds survived'),
    ('invalid', 'invalid actions'),
]


def moving_average(values, window):
    """Centred moving average, shrinking the window at the ends."""
    if window <= 1:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'{path} is empty - has a training run written to it yet?')
    cols = {k: [float(r[k]) for r in rows] for k in rows[0] if r_is_number(rows[0][k])}
    return cols


def r_is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def tidy(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linewidth=0.6)


def draw(ax, x, y, window, label):
    ax.plot(x, y, linewidth=1, color=RAW, label='per report interval')
    ax.plot(x, moving_average(y, window), linewidth=2, color=SERIES,
            label=f'smoothed (window {window})')
    ax.set_title(label, loc='left')
    tidy(ax)


def fig_progress(cols, out, stem, window, fmt):
    """Score and survival, stacked - NOT two y-axes on one plot."""
    x = cols.get('round') or list(range(1, len(next(iter(cols.values()))) + 1))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    draw(a1, x, cols['score'], window, 'score per round')
    if 'survived' in cols:
        draw(a2, x, cols['survived'], window, 'rounds survived')
    a2.set_xlabel('training round')
    a1.legend(loc='upper left', ncol=2)
    fig.suptitle('Training progress', fontsize=11, fontweight='bold', y=0.99)
    fig.tight_layout(h_pad=1.8)
    path = os.path.join(out, f'{stem}_progress.{fmt}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ', path)


def fig_panels(cols, out, stem, window, fmt):
    x = cols.get('round') or list(range(1, len(next(iter(cols.values()))) + 1))
    have = [(k, lab) for k, lab in PANELS if k in cols]
    fig, axes = plt.subplots(len(have), 1, figsize=(6.4, 1.5 * len(have)), sharex=True)
    axes = [axes] if len(have) == 1 else list(axes)
    for ax, (k, lab) in zip(axes, have):
        draw(ax, x, cols[k], window, lab)
    axes[-1].set_xlabel('training round')
    axes[0].legend(loc='upper left', ncol=2)
    fig.suptitle('Training metrics', fontsize=11, fontweight='bold', y=1.0)
    fig.tight_layout(h_pad=1.2)
    path = os.path.join(out, f'{stem}_panels.{fmt}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ', path)


DISCOVERED = '#2a78d6'
FAILED = '#eb6834'


def fig_overlay(runs, out, name, window, metric, fmt, threshold=1.0):
    """Several training runs on one axis, coloured by whether they took off."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    seen = set()
    ends = []
    for label, cols in runs:
        y = moving_average(cols[metric], window)
        x = cols.get('round') or list(range(1, len(y) + 1))
        took_off = cols[metric][-1] > threshold
        col = DISCOVERED if took_off else FAILED
        key = 'discovered bombing' if took_off else 'never discovered bombing'
        ax.plot(x, y, linewidth=2, color=col,
                label=key if key not in seen else None)
        seen.add(key)
        ends.append((x[-1], y[-1], label))

    # end labels, nudged apart when runs converge on the same value
    span = max(e[1] for e in ends) - min(e[1] for e in ends) or 1.0
    ends.sort(key=lambda e: e[1])
    last, stack = None, 0
    for ex, ey, label in ends:
        if last is not None and abs(ey - last) < span * 0.06:
            stack += 1
        else:
            stack = 0
        last = ey
        ax.annotate(label, (ex, ey), xytext=(7, stack * 11),
                    textcoords='offset points',
                    fontsize=7.5, color=INK2, va='center')
    ax.set_xlabel('training round')
    ax.set_ylabel(f'{metric} per round')
    ax.set_title('Fresh training is bimodal', loc='left')
    ax.legend(loc='upper left')
    ax.set_xlim(right=max(max(c.get('round') or [len(c[metric])])
                          for _, c in runs) * 1.14)
    tidy(ax)
    path = os.path.join(out, f'{name}.{fmt}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  ', path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', nargs='+',
                   help='one or more files train.py wrote; several means overlay mode')
    p.add_argument('--smooth', type=int, default=9, help='moving-average window in report intervals')
    p.add_argument('--out', default=None, help='where to write (default: beside the CSV)')
    p.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'])
    p.add_argument('--name', default='overlay', help='output name in overlay mode')
    p.add_argument('--metric', default='score', help='metric to overlay')
    p.add_argument('--threshold', type=float, default=1.0,
                   help='final value above which a run counts as having taken off')
    a = p.parse_args()

    out = a.out or os.path.dirname(os.path.abspath(a.csv[0]))
    os.makedirs(out, exist_ok=True)

    if len(a.csv) > 1:
        runs = [(os.path.splitext(os.path.basename(f))[0], read_csv(f)) for f in a.csv]
        for name, cols in runs:
            print(f'{name}: {len(next(iter(cols.values())))} intervals, '
                  f'final score {cols["score"][-1]:.2f}')
        print('figures:')
        fig_overlay(runs, out, a.name, a.smooth, a.metric, a.format)
        return

    cols = read_csv(a.csv[0])
    stem = os.path.splitext(os.path.basename(a.csv[0]))[0]
    print(f'{len(next(iter(cols.values())))} report intervals, '
          f'columns: {", ".join(sorted(cols))}')
    print('figures:')
    fig_progress(cols, out, stem, a.smooth, a.format)
    fig_panels(cols, out, stem, a.smooth, a.format)


if __name__ == '__main__':
    main()
