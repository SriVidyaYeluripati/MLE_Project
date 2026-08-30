"""
Feature extraction.   Q(s, a) = w . phi(s, a)

Features are ACTION-RELATIVE and share one weight vector: phi describes what an
action *does*, not where we are.  That is why "move toward the nearest coin" is
learned once instead of four times, and why nothing here refers to absolute
board coordinates.

Stage 2 of the roll-out: bombs, crates and survival (task 2).
Stage-1 indices 0-5 are unchanged - later stages append, never renumber.
"""

from collections import deque

import numpy as np

try:
    from .danger import (precompute, danger_map, blocked_map, safe_bfs,
                         blast_coords, MOVES, HORIZON)
except ImportError:
    from danger import (precompute, danger_map, blocked_map, safe_bfs,
                        blast_coords, MOVES, HORIZON)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

FEATURE_NAMES = [
    # --- stage 1: navigation and coins
    'bias',            # 0
    'is_wait',         # 1
    'is_invalid',      # 2
    'd_coin',          # 3   gamma^(steps to nearest reachable coin), afterstate
    'coin_delta',      # 4   -1 closer, 0 same, +1 further
    'no_coin',         # 5
    # --- stage 2: bombs and crates
    'is_bomb',         # 6
    'bomb_ready',      # 7   BOMB is currently legal
    'd_crate',         # 8   gamma^(steps to a tile worth bombing)
    'crate_delta',     # 9
    'crates_hit_1',    # 10  one-hot over crates a bomb here would destroy.
    'crates_hit_2',    # 11  The k = 0 level is deliberately OMITTED: bias plus a
    'crates_hit_3p',   # 12  complete one-hot group is exactly collinear (ch. 2.4)
    # --- stage 2: survival
    'danger_0',        # 13  one-hot: destination becomes lethal in k steps
    'danger_1',        # 14
    'danger_2',        # 15
    'danger_3',        # 16
    'danger_4',        # 17
    'd_safety',        # 18  gamma^(steps from destination to a永 safe tile)
    'safety_delta',    # 19  -1 closer to safety, 0 same, +1 further
    'no_escape',       # 20  no surviving continuation exists from the destination
    'exits',           # 21  free neighbours of the destination / 4
    'dead_end',        # 22  destination has at most one exit
    # --- stage 2: conjunctions a linear model cannot infer
    'x_danger_safety', # 23  in_danger AND moving toward safety
    'x_bomb_noescape', # 24  bombing AND no way out  -> suicide
    'x_bomb_crates2',  # 25  bombing AND it clears two or more crates
    # --- an offset column, constant across actions (see below)
    'phi_state',       # 26  Phi(s) itself
    # --- stage 3: opponents.  Note what is NOT here: the distance LEVEL.
    # Stage 2 showed that level features duplicate the potential and end up
    # absorbing -Phi(s), so opponent distance lives in potential() instead and
    # only the delta and the situation are features.
    'opp_delta',       # 27  -1 closer to the nearest opponent, 0 same, +1 further
    'opp_in_blast',    # 28  a bomb here would cover an opponent's tile
    'x_bomb_opp',      # 29  bombing AND it covers an opponent -> the 5-point play
    'no_opp',          # 30  no opponent reachable
    # --- stage 4: the kill play.  opp_in_blast only says the blast COVERS them;
    # they still have four steps to walk out of it.  opp_trapped asks the
    # question that actually pays: after this bomb, does that opponent have any
    # surviving continuation at all?  Same backward recursion as no_escape.
    'opp_trapped',     # 31  a bomb here leaves an opponent with no escape
    'x_bomb_trapped',  # 32  bombing AND it traps -> this is what a kill looks like
]
N_FEATURES = len(FEATURE_NAMES)
(BIAS, IS_WAIT, IS_INVALID, D_COIN, COIN_DELTA, NO_COIN,
 IS_BOMB, BOMB_READY, D_CRATE, CRATE_DELTA,
 CRATES_1, CRATES_2, CRATES_3P,
 DANGER_0, DANGER_1, DANGER_2, DANGER_3, DANGER_4,
 D_SAFETY, SAFETY_DELTA, NO_ESCAPE, EXITS, DEAD_END,
 X_DANGER_SAFETY, X_BOMB_NOESCAPE, X_BOMB_CRATES2, PHI_STATE,
 OPP_DELTA, OPP_IN_BLAST, X_BOMB_OPP, NO_OPP,
 OPP_TRAPPED, X_BOMB_TRAPPED) = range(N_FEATURES)

GAMMA_FEAT = 0.9
UNREACHABLE = np.iinfo(np.int32).max
BOMB_TIMER = 4          # what a bomb dropped NOW looks like to the danger map
NEIGHBOURS = ((0, -1), (1, 0), (0, 1), (-1, 0))


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #

def multi_source_bfs(sources, blocked):
    """Distance from every tile to the nearest source, ignoring time."""
    dist = np.full(blocked.shape, UNREACHABLE, dtype=np.int32)
    q = deque()
    for s in sources:
        if not blocked[s]:
            dist[s] = 0
            q.append(s)
    while q:
        x, y = q.popleft()
        for dx, dy in NEIGHBOURS:
            v = (x + dx, y + dy)
            if not (0 <= v[0] < blocked.shape[0] and 0 <= v[1] < blocked.shape[1]):
                continue
            if blocked[v] or dist[v] != UNREACHABLE:
                continue
            dist[v] = dist[(x, y)] + 1
            q.append(v)
    return dist


def _dilate(mask):
    """OR of a boolean board with its four neighbours (i.e. 'reachable in one move')."""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def survivable_map(lethal, free):
    """
    surv[j, x, y] : standing on (x, y) at the end of step j, does ANY sequence of
    moves survive to the horizon?  Backward recursion over the time layers, so
    this is exact rather than a distance heuristic - it is what makes no_escape
    trustworthy, and no_escape is the feature that stops the agent killing itself.
    """
    h = lethal.shape[0]
    surv = np.zeros_like(lethal)
    surv[h - 1] = free & ~lethal[h - 1]
    for j in range(h - 2, -1, -1):
        surv[j] = free & ~lethal[j] & _dilate(surv[j + 1])
    return surv


def _shift(a, dx, dy):
    """a shifted so that out[v] == a[v + (dx, dy)], zero-filled at the border."""
    out = np.zeros_like(a)
    xs = slice(max(0, -dx), a.shape[0] - max(0, dx))
    xd = slice(max(0, dx), a.shape[0] - max(0, -dx))
    ys = slice(max(0, -dy), a.shape[1] - max(0, dy))
    yd = slice(max(0, dy), a.shape[1] - max(0, -dy))
    out[xs, ys] = a[xd, yd]
    return out


def bomb_values(field, power=3):
    """
    Crates destroyed by a bomb dropped on each tile, vectorised.
    A blast stops at the first stone wall, so a tile counts only if no wall lies
    between it and the bomb.
    """
    crate = field == 1
    wall = field == -1
    vals = np.zeros(field.shape, dtype=np.int32)
    for dx, dy in NEIGHBOURS:
        blocked = np.zeros(field.shape, dtype=bool)
        for i in range(1, power + 1):
            blocked |= _shift(wall, i * dx, i * dy)
            vals += (_shift(crate, i * dx, i * dy) & ~blocked)
    return vals


# --------------------------------------------------------------------------- #
# per-step context
# --------------------------------------------------------------------------- #

def _danger_view(game_state, extra_bomb=None):
    """lethal / survivable / distance-to-safety, optionally with a hypothetical bomb."""
    gs = game_state
    if extra_bomb is not None:
        gs = dict(game_state)
        gs['bombs'] = list(game_state['bombs']) + [(extra_bomb, BOMB_TIMER)]

    lethal = danger_map(gs)
    free = gs['field'] == 0
    surv = survivable_map(lethal, free)
    never_lethal = free & ~lethal.any(axis=0)
    d_safe = multi_source_bfs(list(zip(*np.nonzero(never_lethal))), ~free)
    return {'lethal': lethal, 'surv': surv, 'd_safe': d_safe}


_CACHE = {}          # tiny content-addressed cache, see context()


def _fingerprint(gs):
    """
    Everything context() depends on - deliberately NOT the step counter.
    The state handed to game_events_occurred as `new_game_state` and the state
    handed to act() on the following step describe the same world, so keying on
    content lets one computation serve both and halves the per-step cost.
    """
    _, _, bombs_left, pos = gs['self']
    return (pos, bombs_left,
            tuple(gs['bombs']), tuple(sorted(gs['coins'])),
            tuple(sorted(o[3] for o in gs['others'])),
            gs['field'].tobytes(), gs['explosion_map'].tobytes())


def context(game_state):
    """Two danger views and three distance maps, shared by all six actions."""
    key = _fingerprint(game_state)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    ctx = _context_uncached(game_state)
    if len(_CACHE) > 4:
        _CACHE.clear()
    _CACHE[key] = ctx
    return ctx


def _context_uncached(game_state):
    _, _, bombs_left, pos = game_state['self']
    ctx = precompute(game_state)
    ctx['bombs_left'] = bool(bombs_left)
    ctx['field'] = game_state['field']

    ctx['now'] = _danger_view(game_state)
    # the BOMB action needs its own view: the bomb we are about to drop
    ctx['after_bomb'] = _danger_view(game_state, extra_bomb=pos) if bombs_left else ctx['now']

    ctx['d_to_coin'] = multi_source_bfs(game_state['coins'], ctx['blocked'])
    ctx['d_coin_here'] = int(ctx['d_to_coin'][pos])

    others = [o[3] for o in game_state['others']]
    free_of_agents = game_state['field'] != 0          # walls and crates only
    ctx['d_to_opp'] = multi_source_bfs(others, free_of_agents) if others else None
    ctx['d_opp_here'] = (int(ctx['d_to_opp'][pos]) if others else UNREACHABLE)
    ctx['others'] = others

    ctx['bomb_value'] = bomb_values(game_state['field'])
    worth_bombing = list(zip(*np.nonzero(ctx['bomb_value'] >= 1)))
    ctx['d_to_crate'] = multi_source_bfs(worth_bombing, ctx['blocked'])
    ctx['d_crate_here'] = int(ctx['d_to_crate'][pos])
    ctx['d_safe_here'] = int(ctx['now']['d_safe'][pos])
    ctx['in_danger'] = bool(ctx['now']['lethal'][:, pos[0], pos[1]].any())
    return ctx


def afterstate(pos, action, blocked, bombs_left):
    """Where we stand after the action, and whether the action was legal."""
    if action in MOVES:
        dx, dy = MOVES[action]
        v = (pos[0] + dx, pos[1] + dy)
        return (pos, False) if blocked[v] else (v, True)
    if action == 'BOMB':
        return pos, bombs_left          # BOMB without a bomb is an INVALID_ACTION
    return pos, True                    # WAIT


def _decay(d):
    return GAMMA_FEAT ** min(int(d), 40) if d != UNREACHABLE else 0.0


def _sign(after, here):
    if after == UNREACHABLE or here == UNREACHABLE:
        return 0.0
    return float(np.sign(after - here))


# --------------------------------------------------------------------------- #
# the feature vector
# --------------------------------------------------------------------------- #

def features(game_state, action, ctx=None):
    if ctx is None:
        ctx = context(game_state)

    phi = np.zeros(N_FEATURES)
    pos = ctx['pos']
    is_bomb = action == 'BOMB'

    phi[BIAS] = 1.0
    phi[IS_WAIT] = float(action == 'WAIT')
    phi[IS_BOMB] = float(is_bomb)
    phi[BOMB_READY] = float(ctx['bombs_left'])

    dest, legal = afterstate(pos, action, ctx['blocked'], ctx['bombs_left'])
    phi[IS_INVALID] = float(not legal)

    # the world as it will be after this action
    view = ctx['after_bomb'] if (is_bomb and ctx['bombs_left']) else ctx['now']

    # ---- coins
    d_here = ctx['d_coin_here']
    if d_here == UNREACHABLE:
        phi[NO_COIN] = 1.0
    else:
        d_after = int(ctx['d_to_coin'][dest])
        # phi[D_COIN] = _decay(d_after)   # level dropped: duplicates Phi(s)
        phi[COIN_DELTA] = _sign(d_after, d_here)

    # ---- crates
    dc_here = ctx['d_crate_here']
    if dc_here != UNREACHABLE:
        dc_after = int(ctx['d_to_crate'][dest])
        # phi[D_CRATE] = _decay(dc_after) # level dropped: duplicates Phi(s)
        phi[CRATE_DELTA] = _sign(dc_after, dc_here)

    if is_bomb and ctx['bombs_left']:
        k = int(ctx['bomb_value'][pos])
        if k == 1:
            phi[CRATES_1] = 1.0
        elif k == 2:
            phi[CRATES_2] = 1.0
        elif k >= 3:
            phi[CRATES_3P] = 1.0

    # ---- survival
    lethal_at = view['lethal'][:, dest[0], dest[1]]
    if lethal_at.any():
        first = int(np.argmax(lethal_at))
        if first <= 4:
            phi[DANGER_0 + first] = 1.0

    ds_after = int(view['d_safe'][dest])
    # phi[D_SAFETY] = _decay(ds_after)  # level dropped: duplicates Phi(s)
    phi[SAFETY_DELTA] = _sign(ds_after, ctx['d_safe_here'])
    phi[NO_ESCAPE] = float(not view['surv'][0][dest])

    free = ctx['field'] == 0
    exits = sum(1 for dx, dy in NEIGHBOURS if free[dest[0] + dx, dest[1] + dy])
    phi[EXITS] = exits / 4.0
    phi[DEAD_END] = float(exits <= 1)

    # ---- opponents
    if ctx['d_to_opp'] is None or ctx['d_opp_here'] == UNREACHABLE:
        phi[NO_OPP] = 1.0
    else:
        phi[OPP_DELTA] = _sign(int(ctx['d_to_opp'][dest]), ctx['d_opp_here'])
        if is_bomb and ctx['bombs_left']:
            hit = set(blast_coords(ctx['field'], pos[0], pos[1]))
            phi[OPP_IN_BLAST] = float(any(o in hit for o in ctx['others']))
            surv = ctx['after_bomb']['surv'][0]
            phi[OPP_TRAPPED] = float(any(o in hit and not surv[o] for o in ctx['others']))

    # ---- conjunctions
    phi[X_DANGER_SAFETY] = float(ctx['in_danger']) * phi[SAFETY_DELTA]
    phi[X_BOMB_NOESCAPE] = phi[IS_BOMB] * phi[NO_ESCAPE]
    phi[X_BOMB_CRATES2] = phi[IS_BOMB] * (phi[CRATES_2] + phi[CRATES_3P])
    phi[X_BOMB_OPP] = phi[IS_BOMB] * phi[OPP_IN_BLAST]
    phi[X_BOMB_TRAPPED] = phi[IS_BOMB] * phi[OPP_TRAPPED]

    # ---- offset column
    # Potential-based shaping is equivalent to initialising the value function
    # to Phi, so what the learner ends up representing is Q*(s,a) - Phi(s).
    # Phi(s) does not depend on the action, but our features are action-relative
    # and correlate strongly with it - so without a dedicated column the learner
    # explains -Phi(s) through d_coin and d_safety and drives THEIR weights
    # negative, which does change the policy.  One constant-across-actions
    # feature absorbs the offset and leaves the others free to do their job.
    phi[PHI_STATE] = potential(game_state, ctx)

    return phi


def feature_matrix(game_state, ctx=None):
    if ctx is None:
        ctx = context(game_state)
    return np.stack([features(game_state, a, ctx) for a in ACTIONS])


def potential(game_state, ctx=None):
    """
    Phi(s) for potential-based shaping - a function of the STATE only, so
    gamma*Phi(s') - Phi(s) cannot change the set of optimal policies and cannot
    pay for a back-and-forth loop.  Phi(terminal) = 0 by construction.
    """
    if game_state is None:
        return 0.0
    if ctx is None:
        ctx = context(game_state)

    phi = 0.25 * _decay(ctx['d_safe_here'])             # be somewhere survivable
    if ctx['d_coin_here'] != UNREACHABLE:
        phi += 0.30 * _decay(ctx['d_coin_here'])        # then chase coins
    elif ctx['d_crate_here'] != UNREACHABLE:
        phi += 0.15 * _decay(ctx['d_crate_here'])       # otherwise open crates
    if ctx['bombs_left'] and ctx['d_opp_here'] != UNREACHABLE:
        phi += 0.25 * _decay(ctx['d_opp_here'])         # hunt only when armed
    return phi


def describe(w, top=None):
    pairs = list(zip(FEATURE_NAMES, w))
    if top:
        pairs = sorted(pairs, key=lambda p: -abs(p[1]))[:top]
    return '  '.join(f'{n}={v:+.3f}' for n, v in pairs)