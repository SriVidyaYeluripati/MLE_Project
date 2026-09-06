"""Adds LQ_BOMBPROBE=1 to model_linearQ/callbacks.py.  Diagnostic only, default off."""
import io, sys, py_compile

p = 'agent_code/model_linearQ/callbacks.py'
s = io.open(p, encoding='utf-8').read()
if 'BOMBPROBE' in s:
    print('already patched'); raise SystemExit

ANCHOR = "    idx = self.rng.choice(len(ACTIONS), p=p)"
if s.count(ANCHOR) != 1:
    raise SystemExit('anchor not found - STOP, tell Claude')

s = s.replace("MASK_INVALID = os.environ.get('LQ_MASK_INVALID', '1') != '0'",
              "MASK_INVALID = os.environ.get('LQ_MASK_INVALID', '1') != '0'\n"
              "BOMBPROBE = os.environ.get('LQ_BOMBPROBE', '0') != '0'   # diagnostic only", 1)

s = s.replace(ANCHOR, ANCHOR + """

    if BOMBPROBE:
        # How many crates does each bomb we drop actually destroy?
        st = self.bombprobe = getattr(self, 'bombprobe', {})
        st['steps'] = st.get('steps', 0) + 1
        if ACTIONS[idx] == 'BOMB':
            try:
                from .danger import blast_coords
            except ImportError:
                from danger import blast_coords
            field = game_state['field']
            bx, by = game_state['self'][3]
            crates = [t for t in blast_coords(field, bx, by) if field[t] == 1]
            st['bombs'] = st.get('bombs', 0) + 1
            st['crates'] = st.get('crates', 0) + len(crates)
            st['n%d' % min(len(crates), 3)] = st.get('n%d' % min(len(crates), 3), 0) + 1
            if crates:
                d = min(abs(cx - bx) + abs(cy - by) for cx, cy in crates)
                st['d%d' % d] = st.get('d%d' % d, 0) + 1
        if st['steps'] % 40000 == 0:
            b = max(st.get('bombs', 0), 1)
            pc = lambda k: 100 * st.get(k, 0) / b
            print('BOMBPROBE bombs=%d  crates/bomb=%.2f  |  0 crates %.0f%%  1 %.0f%%  2 %.0f%%  3+ %.0f%%'
                  '  |  nearest crate at distance 1: %.0f%%  2: %.0f%%  3: %.0f%%'
                  % (st.get('bombs', 0), st.get('crates', 0) / b,
                     pc('n0'), pc('n1'), pc('n2'), pc('n3'),
                     pc('d1'), pc('d2'), pc('d3')), flush=True)""", 1)

io.open(p, 'w', encoding='utf-8').write(s)
py_compile.compile(p, doraise=True)
print('PATCHED - bomb probe added (off unless LQ_BOMBPROBE=1)')