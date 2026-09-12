"""One-off: make training actually persist what it learns.

end_of_round() called save() only on the REPORT_EVERY boundary, and self.round
restarts at 0 on every main.py invocation - so any run shorter than 100 rounds
threw away everything it learned.  Also points the frozen snapshot at the
folder the frozen_linq agent actually reads.
"""
import io, sys, py_compile

t = 'agent_code/model_linearQ/train.py'
s = io.open(t, encoding='utf-8').read()
OLD = """        self.stats = defaultdict(float)
        self.q_max_seen = 0.0
        save(self)
"""
NEW = """        self.stats = defaultdict(float)
        self.q_max_seen = 0.0

    # Save EVERY round, not only on the reporting boundary.  self.round restarts
    # at 0 on each main.py invocation, so any run shorter than REPORT_EVERY
    # rounds never reached the old save() call and silently discarded everything
    # it had learned.  The file is 4 KB; there is no reason to be thrifty.
    save(self)
"""
if NEW in s:
    print('train.py  : already patched')
elif OLD in s:
    io.open(t, 'w', encoding='utf-8').write(s.replace(OLD, NEW, 1))
    py_compile.compile(t, doraise=True)
    print('train.py  : PATCHED (saves every round now)')
else:
    sys.exit('train.py  : pattern not found - stop, do not push, tell Claude')

p = 'setup_pool.sh'
try:
    s = io.open(p, encoding='utf-8').read()
except FileNotFoundError:
    sys.exit(0)
bad, good = 'cp $A/weights.npz $A/frozen.npz', 'cp $A/weights.npz agent_code/frozen_linq/frozen.npz'
if good in s:
    print('setup_pool: already patched')
elif bad in s:
    io.open(p, 'w', encoding='utf-8').write(s.replace(bad, good, 1))
    print('setup_pool: PATCHED (frozen.npz now goes to frozen_linq/)')
else:
    print('setup_pool: nothing to change')
