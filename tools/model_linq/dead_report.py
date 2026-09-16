"""Print every LQ_* experiment switch in the agent files, with line numbers.

Run from the repo root:  python tools/model_linq/dead_report.py
Everything it prints is experiment scaffolding that does not run in a
tournament game.  Use it as a checklist while rewriting.
"""
import re, sys, os

FILES = ['callbacks.py', 'features.py', 'train.py', 'danger.py']
AGENT = os.path.join('agent_code', 'model_linearQ')

ENV = re.compile(r"os\.environ\.get\(\s*['\"](LQ_[A-Z0-9_]+)['\"]")
FLAG = re.compile(r"^\s*(PHANTOM|HERD|PHASE|RFF|ABLATED|FIX_TRAPPED|BOMB_TARGET|"
                  r"SAFE_FILTER|BOMB_FILTER|PROBE|BOMBPROBE|COINTRACE|KILLTRACE|"
                  r"HERDPROBE|PHASEPROBE|BCLOG|RACE_FILTER|WSCALE|TAU_PLAY)\b")

def indent(line):
    return len(line) - len(line.lstrip())

for fn in FILES:
    path = os.path.join(AGENT, fn)
    if not os.path.exists(path):
        continue
    src = open(path).read().splitlines()
    print('=' * 60)
    print(path, '--', len(src), 'lines')
    print('=' * 60)

    # 1. module-level switch definitions
    print('\n-- switch definitions (delete, hard-code the value) --')
    for i, line in enumerate(src, 1):
        m = ENV.search(line)
        if m:
            print(f'  {i:4d}  {line.strip()[:78]}')

    # 2. blocks guarded by a switch
    print('\n-- guarded blocks (delete whole block) --')
    i = 0
    while i < len(src):
        line = src[i]
        if re.match(r'^\s*if\s+', line) and FLAG.search(line.split('if', 1)[1]):
            start = i + 1
            base = indent(line)
            j = i + 1
            while j < len(src):
                nxt = src[j]
                if nxt.strip() and indent(nxt) <= base:
                    if re.match(r'^\s*(elif|else)\b', nxt) and indent(nxt) == base:
                        j += 1
                        continue
                    break
                j += 1
            print(f'  {start:4d}-{j:<4d} ({j - start + 1:3d} lines)  {line.strip()[:60]}')
            i = j
        else:
            i += 1

    # 3. probe functions
    print('\n-- probe functions (delete whole function) --')
    for i, line in enumerate(src, 1):
        if re.match(r'^def _(cointrace|killtrace|herdprobe|phaseprobe|bclog)', line) \
           or re.match(r'^def apply_wscale', line):
            print(f'  {i:4d}  {line.strip()}')

    # 4. the import dance
    n = sum(1 for k in range(len(src) - 1)
            if src[k].strip() == 'try:' and 'from .' in src[k + 1])
    print(f'\n-- try/except relative-import pairs: {n} '
          f'(replace each with the plain "from .x import y" line)')
    print()
