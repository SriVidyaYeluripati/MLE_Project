"""
Always loaded, including in the tournament.   Keep this file fast and free of
any training-only imports: train.py does not exist at tournament time.
"""

import os

import numpy as np

try:
    from .features import (ACTIONS, N_FEATURES, BIAS, NO_ESCAPE, IS_INVALID,
                           IS_BOMB, CRATES_1, CRATES_2, CRATES_3P, OPP_TRAPPED,
                           OPP_IN_BLAST, feature_matrix, context)
except ImportError:
    from features import (ACTIONS, N_FEATURES, BIAS, NO_ESCAPE, IS_INVALID,
                          IS_BOMB, CRATES_1, CRATES_2, CRATES_3P, OPP_TRAPPED,
                          OPP_IN_BLAST, feature_matrix, context)

# LQ_WEIGHTS lets an ablation run write its own file instead of clobbering the
# shipped weights.  Relative to THIS file - never absolute.
MODEL_FILE = os.environ.get('LQ_WEIGHTS', 'weights.npz')

OPTIMISTIC_INIT = 0.5               # untried actions look attractive (ch. 4)
TAU_TRAIN = 0.25                    # softmax temperature while training
_tp = os.environ.get('LQ_TAU_PLAY', '')
TAU_PLAY = float(_tp) if _tp.strip() != '' else 0.0   # near-greedy when it counts
# chbridges (Heidelberg, WS20/21) report that their agent scored BETTER at a
# deployment epsilon of 0.25 than at 0.05, because pure greedy oscillates
# between two tiles.  Model LinQ ships greedy.  LQ_TAU_PLAY makes that
# testable without retraining - it is a policy change, not a weight change.
TIE_EPS = 1e-9                      # values this close count as tied

# Exploration is NOT uniform over the six actions: a random BOMB can end the
# episode, and an episode that ends early stops producing data (ch. 4.2).
BOMB_EXPLORE = float(os.environ.get('LQ_BOMB_EXPLORE', 0.25))
                                    # BOMB sampled at this fraction of its softmax
                                    # share. Damping protects an agent that bombs
                                    # too much - but an agent that has not yet
                                    # DISCOVERED bombing needs the opposite, and
                                    # 2 of 3 fresh trainings never discover it.
MASK_FATAL_ROUNDS = 0               # curriculum: mask provably-fatal actions for the
                                    # first N training rounds, then learn from real
                                    # deaths. 0 = off. Ablate this.

# An illegal action is not a mistake the agent has to LEARN about: it is a move
# into a wall, or BOMB with no bomb.  The framework leaves the agent where it
# stood, so its afterstate is identical to WAIT's - the two differ only by the
# columns is_wait and is_invalid.  Whenever the learner ends up with
#     w[is_invalid] > w[is_wait]
# an illegal move becomes a strictly cheaper way to wait, and a *greedy* policy
# will then take one every time waiting is right.  That is why the invalid
# count barely moved when we went greedy: those actions were never exploration.
# Masking removes the whole failure mode instead of hoping the weights order
# themselves correctly (Huang & Ontanon 2022, invalid action masking).
MASK_INVALID = os.environ.get('LQ_MASK_INVALID', '1') != '0'    # ablate this.

# The same argument, one step further.  no_escape is EXACT, not a heuristic: it
# is the backward recursion over the bomb timeline in danger.py, and it says the
# destination has no surviving continuation at all.  Leaving that to the linear
# model means a large enough Q elsewhere can outvote a certain death, and the
# agent kills itself 0.45 times per round.  A hard filter removes the choice.
# This is the "ActionFilter" of the Skynet Pommerman agent (Gao et al. 2019) and
# the shallow-lookahead safety filter of Kartal et al. (2019), both of which
# report it as the component that makes model-free RL viable in Bomberman.
SAFE_FILTER = os.environ.get('LQ_SAFE_FILTER', '0') != '0'

# The measured weakness, stated as a number: against three coin_collector_agents
# the agent destroys 0.46 crates per bomb and they destroy 2.87.  The learned
# weights say why - is_bomb carries an UNCONDITIONAL +0.035, while crates_hit_1
# is +0.007 and crates_hit_2 is -0.005.  Bombing is attractive in itself, so the
# agent pays the four-step retreat over and over for nothing.  This filter
# forbids only the bombs that provably accomplish nothing: no crate in the
# blast and no opponent trapped by it.
BOMB_FILTER = os.environ.get('LQ_BOMB_FILTER', '0') != '0'
PROBE = os.environ.get('LQ_PROBE', '0') != '0'
BOMBPROBE = os.environ.get('LQ_BOMBPROBE', '0') != '0'
COINTRACE = os.environ.get('LQ_COINTRACE', '0') != '0'
KILLTRACE = os.environ.get('LQ_KILLTRACE', '0') != '0'
HERDPROBE = os.environ.get('LQ_HERDPROBE', '0') != '0'
PHASEPROBE = os.environ.get('LQ_PHASEPROBE', '0') != '0'
_PSTAT = {'late': [], 'spread': [], 'n': 0, 'varies': 0}
_HSTAT = {}
_rf = os.environ.get('LQ_RACE_FILTER', '')
RACE_FILTER = int(_rf) if _rf.strip() != '' else None
_RSTAT = {}

# Play-time weight surgery, for asking "is this feature's pull too weak?"
# without retraining.  LQ_WSCALE='coin_delta=2,crate_delta=-0.05' multiplies the
# named weight (or SETS it, when the value is written with a leading '=').
# Diagnostic only; the shipped agent sets nothing.
WSCALE = os.environ.get('LQ_WSCALE', '')


def apply_wscale(w, names, logger=None):
    if not WSCALE:
        return w
    w = w.copy()
    for item in WSCALE.split(','):
        key, _, val = item.partition('=')
        key = key.strip()
        if key not in names:
            raise KeyError(f'LQ_WSCALE: unknown feature {key!r}')
        i = names.index(key)
        if val.startswith('='):
            w[i] = float(val[1:])
        else:
            w[i] *= float(val)
        if logger:
            logger.info(f'LQ_WSCALE {key}: -> {w[i]:+.4f}')
    return w


def model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)


def setup(self):
    """Called once before the first round."""
    path = model_path()
    self.rng = np.random.default_rng()

    if os.path.isfile(path):
        data = np.load(path, allow_pickle=True)
        w = data['w'].astype(float)
        if len(w) < N_FEATURES:
            # New features were appended.  Keep everything already learned and
            # start the new columns at zero: a warm start is worth ~900 rounds
            # of phase 1 and avoids the bimodal fresh-training failure entirely.
            self.logger.warning(f'weights have {len(w)} of {N_FEATURES} features'
                                f' - padding the new ones with zeros (warm start)')
            w = np.concatenate([w, np.zeros(N_FEATURES - len(w))])
        elif len(w) > N_FEATURES:
            raise ValueError(f'{MODEL_FILE} has {len(w)} features, code has {N_FEATURES}')
        self.w = w
        self.w = apply_wscale(self.w, [str(x) for x in data['feature_names']],
                              self.logger)
        self.logger.info(f'loaded weights from {MODEL_FILE}')
    elif getattr(self, 'train', False):
        self.w = np.zeros(N_FEATURES)
        self.w[BIAS] = OPTIMISTIC_INIT
        self.logger.info('no weights found - starting from optimistic init')
    else:
        raise FileNotFoundError(
            f'{MODEL_FILE} is missing; train the agent before playing')

    self.tau = TAU_TRAIN if getattr(self, 'train', False) else TAU_PLAY

    # The ablation switches are environment variables, so a stale export in the
    # shell would silently ship a crippled agent - LQ_ABLATE=escape costs -5.30
    # margin and changes nothing visible on screen.  One log line at startup
    # makes any such accident findable afterwards instead of unexplained.
    try:
        from .features import ABLATED
    except ImportError:
        from features import ABLATED
    self.logger.info(
        f'config: tau={self.tau} mask_invalid={MASK_INVALID} '
        f'safe_filter={SAFE_FILTER} bomb_filter={BOMB_FILTER} '
        f'weights={MODEL_FILE} ablated={ABLATED or "none"}')
    if ABLATED:
        self.logger.warning(f'FEATURES ABLATED: {ABLATED} - not the shipped agent')


def legal_mask(phi):
    """Rows of a feature matrix whose action the framework will actually run."""
    return phi[:, IS_INVALID] < 0.5


def playable_mask(phi, ctx=None):
    """
    legal_mask, optionally intersected with "does not provably kill us".

    Falls back to the legal mask whenever every legal action is fatal, so the
    filter can never empty the action set - in that position the agent is dead
    whatever it does, and the weights may as well choose how.
    """
    m = legal_mask(phi)

    # ---- the race gate.  Section 11 measured that opening crates efficiently
    # gains nothing, because in classic the coins are sealed inside them and
    # every crate opened is a coin offered to whoever is nearest.  This gate
    # asks the question the bomb filter does not: not "does this bomb hit
    # something" but "will I still be here to collect what it releases".
    #
    # A bomb dropped now detonates at t+4 and its fire clears at t+6, so a coin
    # it releases is collectable from t+6.  If an opponent is within K steps of
    # this tile, they can be standing on that coin before the agent - which has
    # to retreat from its own blast first - gets back.  K defaults to that full
    # 6-step lifecycle.  A bomb that traps an opponent is never blocked: a kill
    # is worth 5 points and is not a contested resource.
    if RACE_FILTER is not None and ctx is not None and ctx.get('d_to_opp') is not None:
        _RSTAT['steps'] = _RSTAT.get('steps', 0) + 1
        d_opp = int(ctx['d_to_opp'][ctx['pos']])
        if d_opp <= RACE_FILTER:
            contested = m & ((phi[:, IS_BOMB] < 0.5) | (phi[:, OPP_TRAPPED] > 0.5))
            if contested.any():
                if not np.array_equal(contested, m):
                    _RSTAT['blocked'] = _RSTAT.get('blocked', 0) + 1
                m = contested
        if _RSTAT['steps'] % 2000 == 0:
            import json
            with open(os.environ.get('LQ_RACE_OUT', '/tmp/racefilter.json'), 'w') as f:
                json.dump(_RSTAT, f)

    if SAFE_FILTER:
        survivable = m & (phi[:, NO_ESCAPE] < 0.5)
        if survivable.any():
            m = survivable

    if BOMB_FILTER:
        hits = (phi[:, CRATES_1] + phi[:, CRATES_2] + phi[:, CRATES_3P]
                + phi[:, OPP_TRAPPED]) > 0.5
        useful = m & (hits | (phi[:, IS_BOMB] < 0.5))
        if useful.any():
            m = useful

    return m


def action_probabilities(q, tau, legal=None):
    """
    Scale-aware softmax over the LEGAL actions.  A fixed temperature is NOT
    scale free: early in training all Q-values sit near zero (near-uniform) and
    late they are far apart (near-greedy), so the exploration rate would anneal
    itself in a way we never chose.  Dividing by the spread fixes that.

    `legal` is a boolean mask; illegal actions get probability 0 and are left
    out of both the max and the spread, so masking cannot silently rescale the
    temperature.  WAIT is always legal, so the mask is never empty.
    """
    q = np.asarray(q, dtype=float)
    if legal is None or not MASK_INVALID:
        legal = np.ones(len(q), dtype=bool)
    else:
        legal = np.asarray(legal, dtype=bool)
        if not legal.any():
            legal = np.ones(len(q), dtype=bool)

    q_legal = q[legal]
    p = np.zeros(len(q))

    if tau <= 0:                                     # greedy, ties shared
        p[legal & (q >= q_legal.max() - TIE_EPS)] = 1.0
        return p / p.sum()

    spread = q_legal.std()
    if spread < 1e-8:
        p[legal] = 1.0
        return p / p.sum()
    z = (q - q_legal.max()) / (tau * spread)
    p[legal] = np.exp(z[legal])
    return p / p.sum()


def act(self, game_state: dict) -> str:
    ctx = context(game_state)                        # one BFS, reused six times
    phi = feature_matrix(game_state, ctx)            # (|A|, N_FEATURES)
    q = phi @ self.w

    # cached for train.py so the update never recomputes what act() already knows
    self.last_phi, self.last_q, self.last_ctx = phi, q, ctx

    mask = playable_mask(phi, ctx)
    p = action_probabilities(q, self.tau, mask)

    if PROBE:                       # diagnostic only: how often does the filter bite?
        base = action_probabilities(q, self.tau, legal_mask(phi))
        self.probe_steps = getattr(self, 'probe_steps', 0) + 1
        if int(np.argmax(base)) != int(np.argmax(p)):
            self.probe_changed = getattr(self, 'probe_changed', 0) + 1
        if self.probe_steps % 20000 == 0:
            ch = getattr(self, 'probe_changed', 0)
            print(f'PROBE steps={self.probe_steps} changed={ch} '
                  f'({100*ch/self.probe_steps:.2f}%)', flush=True)

    if getattr(self, 'train', False):
        # damp exploratory self-destruction; the greedy choice is untouched
        p = p.copy()
        p[ACTIONS.index('BOMB')] *= BOMB_EXPLORE
        if game_state['round'] <= MASK_FATAL_ROUNDS:
            survivable = phi[:, NO_ESCAPE] < 0.5
            if (p * survivable).sum() > 0:
                p *= survivable
        p /= p.sum()

    idx = self.rng.choice(len(ACTIONS), p=p)

    if BOMBPROBE:
        st = self.bombprobe = getattr(self, 'bombprobe', {})
        st['steps'] = st.get('steps', 0) + 1
        if ACTIONS[idx] == 'BOMB':
            try:
                from .danger import blast_coords
            except ImportError:
                from danger import blast_coords
            field = game_state['field']
            bx, by = game_state['self'][3]
            hit = blast_coords(field, bx, by)
            crates = [t for t in hit if field[t] == 1]
            st['bombs'] = st.get('bombs', 0) + 1
            st['crates'] = st.get('crates', 0) + len(crates)
            n = min(len(crates), 3)
            st[f'n{n}'] = st.get(f'n{n}', 0) + 1
            if crates:
                d = min(abs(cx - bx) + abs(cy - by) for cx, cy in crates)
                st[f'd{d}'] = st.get(f'd{d}', 0) + 1
        if st['steps'] % 40000 == 0:
            b = max(st.get('bombs', 0), 1)
            print('BOMBPROBE bombs=%d  crates/bomb=%.2f  |  0 crates %.0f%%  1 %.0f%%  2 %.0f%%  3+ %.0f%%'
                  '  |  nearest crate at distance 1: %.0f%%  2: %.0f%%  3: %.0f%%'
                  % (st.get('bombs', 0), st.get('crates', 0)/b,
                     100*st.get('n0', 0)/b, 100*st.get('n1', 0)/b,
                     100*st.get('n2', 0)/b, 100*st.get('n3', 0)/b,
                     100*st.get('d1', 0)/b, 100*st.get('d2', 0)/b, 100*st.get('d3', 0)/b),
                  flush=True)
    if COINTRACE:
        _cointrace(self, ctx, phi, mask, ACTIONS[idx])

    if KILLTRACE:
        _killtrace(self, ctx, phi, mask, ACTIONS[idx])

    if PHASEPROBE:
        try:
            import json as _j
            try:
                from .features import X_LATE_OPPDELTA, X_LATE_BOMB, X_LATE_BOMBOPP
            except ImportError:
                from features import X_LATE_OPPDELTA, X_LATE_BOMB, X_LATE_BOMBOPP
            st = _PSTAT
            st['n'] += 1
            st['late'].append(round(float(ctx.get('late', 0.0)), 3))
            cols = phi[:, [X_LATE_OPPDELTA, X_LATE_BOMB, X_LATE_BOMBOPP]]
            sp = float(cols.max(axis=0).max() - cols.min(axis=0).min())
            st['spread'].append(round(sp, 3))
            if sp > 1e-9:
                st['varies'] += 1
            if st['n'] % 500 == 0:
                import numpy as _np
                L = _np.array(st['late'])
                _j.dump(dict(n=st['n'], varies=st['varies'],
                             late_mean=float(L.mean()), late_p50=float(_np.median(L)),
                             late_p90=float(_np.percentile(L, 90)), late_max=float(L.max()),
                             frac_late_over_0_5=float((L > 0.5).mean()),
                             frac_late_over_0_8=float((L > 0.8).mean())),
                        open('/tmp/phaseprobe.json', 'w'), indent=1)
        except Exception:
            import traceback
            open('/tmp/phaseprobe_err.txt', 'w').write(traceback.format_exc())

    if HERDPROBE:
        try:
            _herdprobe(self, ctx, phi, ACTIONS[idx])
        except Exception:
            import traceback
            open('/tmp/herdprobe_err.txt', 'w').write(traceback.format_exc())

    self.logger.debug(f'q={np.round(q, 3)} -> {ACTIONS[idx]}')
    return ACTIONS[idx]


def _cointrace(self, ctx, phi, mask, chosen):
    """
    Diagnostic: where is the coin race actually lost?

    On every step where a coin is reachable AND some legal action strictly
    decreases the BFS distance to it, the agent had a chance to close the gap.
    Count how often it took that chance and what it did instead.  Results go
    to a file because the agent runs in a subprocess whose stdout is dropped.
    Nothing here influences play.
    """
    try:
        from .features import COIN_DELTA, UNREACHABLE
    except ImportError:
        from features import COIN_DELTA, UNREACHABLE
    st = self.cointrace = getattr(self, 'cointrace', {})
    st['steps'] = st.get('steps', 0) + 1

    d_here = int(ctx['d_coin_here'])
    if d_here != UNREACHABLE:
        st['coin_visible'] = st.get('coin_visible', 0) + 1
        st['d_sum'] = st.get('d_sum', 0) + d_here
        closer = [a for i, a in enumerate(ACTIONS)
                  if mask[i] and phi[i, COIN_DELTA] < 0]
        if not closer:
            st['boxed_in'] = st.get('boxed_in', 0) + 1
        else:
            st['could_approach'] = st.get('could_approach', 0) + 1
            if chosen in closer:
                st['did_approach'] = st.get('did_approach', 0) + 1
            else:
                st['miss_danger' if ctx['in_danger'] else 'miss_free'] = \
                    st.get('miss_danger' if ctx['in_danger'] else 'miss_free', 0) + 1
                st['miss_' + chosen] = st.get('miss_' + chosen, 0) + 1

    if st['steps'] % 1000 == 0:
        import json
        with open(os.environ.get('LQ_COINTRACE_OUT', '/tmp/cointrace.json'), 'w') as f:
            json.dump(st, f)

def _killtrace(self, ctx, phi, mask, chosen):
    """
    Diagnostic: is the kill channel starved of OPPORTUNITY, or of ACTION?

    A kill is worth 5 points and Model LinQ takes only 0.11 per round, while
    the whole deficit against coin_collector is 0.04 kills.  Two very different
    explanations, and they call for opposite fixes:

      opportunity-starved  the situation almost never arises, so no weight on
                           opp_trapped can help - the agent has to be made to
                           CREATE the situation (positioning), not to value it
      action-starved       the situation arises and the policy declines it, so
                           a weight or a filter is enough

    This counts both.  Nothing here influences play.
    """
    b = ACTIONS.index('BOMB')
    st = self.killtrace = getattr(self, 'killtrace', {})
    st['steps'] = st.get('steps', 0) + 1

    if not mask[b]:
        st['bomb_illegal'] = st.get('bomb_illegal', 0) + 1
    else:
        st['bomb_legal'] = st.get('bomb_legal', 0) + 1
        covers = phi[b, OPP_IN_BLAST] > 0.5
        traps = phi[b, OPP_TRAPPED] > 0.5
        if covers:
            st['covers'] = st.get('covers', 0) + 1
            if chosen == 'BOMB':
                st['covers_took'] = st.get('covers_took', 0) + 1
        if traps:
            st['traps'] = st.get('traps', 0) + 1
            if chosen == 'BOMB':
                st['traps_took'] = st.get('traps_took', 0) + 1
            else:
                st['traps_missed_' + chosen] = st.get('traps_missed_' + chosen, 0) + 1

    if st['steps'] % 1000 == 0:
        import json
        with open(os.environ.get('LQ_KILLTRACE_OUT', '/tmp/killtrace.json'), 'w') as f:
            json.dump(st, f)


def _herdprobe(self, ctx, phi, chosen):
    """
    Redundancy check for the proposed herding feature, run BEFORE training.
    See experiments/herd_probe.py for the reasoning; the maths lives there so
    the two cannot drift apart.
    """
    import json, sys, os as _o
    sys.path.insert(0, _o.path.join(_o.path.dirname(__file__), '..', '..', '..', 'experiments', 'model_linq'))
    from herd_probe import confine, NEAR
    try:
        from .features import OPP_DELTA, EXITS, afterstate
    except ImportError:
        from features import OPP_DELTA, EXITS, afterstate

    st = _HSTAT
    st['steps'] = st.get('steps', 0) + 1
    if st['steps'] % 200 == 0:            # before any early return, or it never runs
        json.dump(st, open('/tmp/herdprobe.json', 'w'), indent=1)
    if ctx.get('d_to_opp') is None:
        return
    d_here = int(ctx['d_to_opp'][ctx['pos']])
    if d_here > NEAR:
        return
    st['near'] = st.get('near', 0) + 1

    others = ctx['others']
    opp = min(others, key=lambda o: abs(o[0]-ctx['pos'][0]) + abs(o[1]-ctx['pos'][1]))
    field = ctx['field']

    MOVES = [0, 1, 2, 3]                       # UP RIGHT DOWN LEFT in ACTIONS
    vals, legal = [], []
    for a in MOVES:
        dest, ok = afterstate(ctx['pos'], ACTIONS[a], ctx['blocked'], ctx['bombs_left'])
        legal.append(ok)
        vals.append(confine(dest, field, opp) if ok else -1.0)
    vals = np.array(vals); legal = np.array(legal)
    if legal.sum() < 2:
        return
    st['usable'] = st.get('usable', 0) + 1

    v = vals[legal]
    if v.max() - v.min() < 1e-9:
        st['flat'] = st.get('flat', 0) + 1       # Q1: no gradient at all
        return
    st['varies'] = st.get('varies', 0) + 1

    # Q2 -- does its best move differ from opp_delta's best move?
    od = phi[MOVES, OPP_DELTA]
    best_new = {i for i, a in enumerate(MOVES) if legal[i] and vals[i] >= v.max() - 1e-9}
    od_l = od[legal]
    best_od = {i for i, a in enumerate(MOVES) if legal[i] and od[i] <= od_l.min() + 1e-9}
    if best_new & best_od:
        st['agree_oppdelta'] = st.get('agree_oppdelta', 0) + 1

    # Q3 -- does it differ from `exits`, which describes MY destination?
    ex = phi[MOVES, EXITS]
    ex_l = ex[legal]
    best_ex = {i for i, a in enumerate(MOVES) if legal[i] and ex[i] <= ex_l.min() + 1e-9}
    if best_new & best_ex:
        st['agree_exits'] = st.get('agree_exits', 0) + 1

