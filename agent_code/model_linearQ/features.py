#Vidya's part
from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MOVES = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}
NEIGHBOURS = ((0, -1), (1, 0), (0, 1), (-1, 0))

BOMB_POWER = 3
EXPLOSION_TIMER = 2
BOMB_TIMER = 4
HORIZON = 7

GAMMA_FEAT = 0.9
UNREACHABLE = np.iinfo(np.int32).max


def blast_coords(field, bx, by, power=BOMB_POWER):
    #Here tiles are covered by a bomb at (bx, by). Mirrors Bomb.get_blast_coords.
    coords = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            x, y = bx + i * dx, by + i * dy
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
            if field[x, y] == -1:
                break
            coords.append((x, y))
    return coords


def danger_map(game_state, horizon=HORIZON):
# We would want to use the explosion_map, but it is not updated for bombs that are just dropped :(.
# But we can compute by using the blast coordinates of all bombs, and the explosion_map for beginning step.
    field = game_state['field']
    lethal = np.zeros((horizon,) + field.shape, dtype=bool)

    for (bx, by), t in game_state['bombs']:
        for (x, y) in blast_coords(field, bx, by):
            for j in (t, t + 1):
                if j < horizon:
                    lethal[j, x, y] = True

    exp = game_state['explosion_map']
    lethal[0][exp >= 1] = True
    
    # array is read only, so we can use it as a cache key.
    lethal.flags.writeable = False

    return lethal


def blocked_map(game_state):
# We make sure that obviously tiles we cannot move (bombs,crates,walls or other agents) onto.
    field = game_state['field']
    blocked = field != 0
    for (bx, by), _ in game_state['bombs']:
        blocked[bx, by] = True
    for _, _, _, (ox, oy) in game_state['others']:
        blocked[ox, oy] = True
    return blocked


def safe_bfs(start, blocked, lethal, horizon=HORIZON):
# Normal BFS would be a bit simpler, but we need to avoid lethal tiles at the right time step., so we use Time aware BFS
    dist = np.full(blocked.shape, UNREACHABLE, dtype=np.int32)
    dist[start] = 0

    frontier = {start}
    reach = []
    #so we can see which tiles are reachable at each time step and also the distance to each tile.
    for j in range(horizon):
        nxt = set()
        for (x, y) in frontier:
            for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
                v = (x + dx, y + dy)
                if not (0 <= v[0] < blocked.shape[0] and 0 <= v[1] < blocked.shape[1]):
                    continue
                if (dx, dy) != (0, 0) and blocked[v]:
                    continue
                if lethal[j][v]:
                    continue
                nxt.add(v)
                if dist[v] > j + 1:
                    dist[v] = j + 1
        reach.append(nxt)
        
        frontier = nxt
        if not frontier:
            reach.extend(set() for _ in range(horizon - 1 - j))
            break

    return reach, dist


def precompute(game_state,lethal=None):
   #One danger map and one BFS per step, reused by all six actions.
    _, _, _, pos = game_state['self']
    if lethal is None:
        lethal = danger_map(game_state)
    blocked = blocked_map(game_state)
    reach, dist = safe_bfs(pos, blocked, lethal)
    return {
        'pos': pos,
        'lethal': lethal,
        'blocked': blocked,
        'reach': reach,
        'dist': dist,
        'escape_exists': bool(reach[-1]),
    }


def multi_source_bfs(sources, blocked):
#Distance from every tile to the nearest source, ignoring time.
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
#OR of a boolean board with its four neighbours ('reachable in one move').
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def survivable_map(lethal, free):
    #surv[j, x, y]: from (x, y) at end of step j, does ANY move sequence survive?
    h = lethal.shape[0]
    surv = np.zeros_like(lethal)
    surv[h - 1] = free & ~lethal[h - 1]
    for j in range(h - 2, -1, -1):
        surv[j] = free & ~lethal[j] & _dilate(surv[j + 1])
    return surv


def _shift(a, dx, dy):
    #a shifted so that out[v] == a[v + (dx, dy)], zero-filled at the border.
    out = np.zeros_like(a)
    xs = slice(max(0, -dx), a.shape[0] - max(0, dx))
    xd = slice(max(0, dx), a.shape[0] - max(0, -dx))
    ys = slice(max(0, -dy), a.shape[1] - max(0, dy))
    yd = slice(max(0, dy), a.shape[1] - max(0, -dy))
    out[xs, ys] = a[xd, yd]
    return out


def bomb_values(field, power=BOMB_POWER):
    #Crates destroyed by a bomb dropped on each tile.
    crate = field == 1
    wall = field == -1
    vals = np.zeros(field.shape, dtype=np.int32)
    for dx, dy in NEIGHBOURS:
        blocked = np.zeros(field.shape, dtype=bool)
        for i in range(1, power + 1):
            blocked |= _shift(wall, i * dx, i * dy)
            vals += (_shift(crate, i * dx, i * dy) & ~blocked)
    return vals


#DO NOT DELETE THEM! because weights.npz is indexed by POSITION. Renumbering silently
# reassigns every learned weight after index 3.
FEATURE_NAMES = [
    'bias',
    'is_wait',
    'is_invalid',
    'd_coin',
    'coin_delta',
    'no_coin',
    'is_bomb',
    'bomb_ready',
    'd_crate',
    'crate_delta',
    'crates_hit_1',
    'crates_hit_2',
    'crates_hit_3p',
    'danger_0',
    'danger_1',
    'danger_2',
    'danger_3',
    'danger_4',
    'd_safety',
    'safety_delta',
    'no_escape',
    'exits',
    'dead_end',
    'x_danger_safety',
    'x_bomb_noescape',
    'x_bomb_crates2',
    'phi_state',
    'opp_delta',
    'opp_in_blast',
    'x_bomb_opp',
    'no_opp',
    'opp_trapped',
    'x_bomb_trapped',
    'wincoin_delta',
    'no_wincoin',
    'mycrate_delta',
]
N_FEATURES = len(FEATURE_NAMES)
(BIAS, IS_WAIT, IS_INVALID, D_COIN, COIN_DELTA, NO_COIN,
 IS_BOMB, BOMB_READY, D_CRATE, CRATE_DELTA,
 CRATES_1, CRATES_2, CRATES_3P,
 DANGER_0, DANGER_1, DANGER_2, DANGER_3, DANGER_4,
 D_SAFETY, SAFETY_DELTA, NO_ESCAPE, EXITS, DEAD_END,
 X_DANGER_SAFETY, X_BOMB_NOESCAPE, X_BOMB_CRATES2, PHI_STATE,
 OPP_DELTA, OPP_IN_BLAST, X_BOMB_OPP, NO_OPP,
 OPP_TRAPPED, X_BOMB_TRAPPED,
 WINCOIN_DELTA, NO_WINCOIN, MYCRATE_DELTA) = range(N_FEATURES)


W_HUNT = 0.25
W_CRATE = 0.15

def _danger_view(game_state, extra_bomb=None,lethal=None):
    #lethal or survivable  or distance-to-safety, optionally with a hypothetical bomb.
    gs = game_state
    if extra_bomb is not None:
        gs = dict(game_state)
        gs['bombs'] = list(game_state['bombs']) + [(extra_bomb, BOMB_TIMER)]
    
    if lethal is None:
        lethal = danger_map(gs)

    free = gs['field'] == 0
    for _, _, _, (ox, oy) in gs['others']:
        free[ox, oy] = False
    surv = survivable_map(lethal, free)
    never_lethal = free & ~lethal.any(axis=0)
    d_safe = multi_source_bfs(list(zip(*np.nonzero(never_lethal))), ~free)
    return {'lethal': lethal, 'surv': surv, 'd_safe': d_safe}

_CACHE = {}


def _fingerprint(gs):
#Everything context() depends on - deliberately NOT the step counter.
    _, _, bombs_left, pos = gs['self']
    return (pos, bombs_left,
            tuple(gs['bombs']), tuple(sorted(gs['coins'])),
            tuple(sorted(o[3] for o in gs['others'])),
            gs['field'].tobytes(), gs['explosion_map'].tobytes())


def context(game_state):
#Two danger views and several distance maps, shared by all six actions.
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

    lethal_now = danger_map(game_state)
    ctx = precompute(game_state, lethal=lethal_now)
    ctx['bombs_left'] = bool(bombs_left)
    ctx['field'] = game_state['field']

    ctx['now'] = _danger_view(game_state, lethal=lethal_now)
    ctx['after_bomb'] = _danger_view(game_state, extra_bomb=pos) if bombs_left else ctx['now']

    ctx['d_to_coin'] = multi_source_bfs(game_state['coins'], ctx['blocked'])
    ctx['d_coin_here'] = int(ctx['d_to_coin'][pos])

    others = [o[3] for o in game_state['others']]
    free_of_agents = game_state['field'] != 0
    ctx['d_to_opp'] = multi_source_bfs(others, free_of_agents) if others else None
    ctx['d_opp_here'] = (int(ctx['d_to_opp'][pos]) if others else UNREACHABLE)
    ctx['others'] = others

    ctx['bomb_value'] = bomb_values(game_state['field'])
    worth_bombing = list(zip(*np.nonzero(ctx['bomb_value'] >= 1)))
    ctx['d_to_crate'] = multi_source_bfs(worth_bombing, ctx['blocked'])
    ctx['d_crate_here'] = int(ctx['d_to_crate'][pos])

    ctx['d_from_me'] = multi_source_bfs([pos], ctx['blocked'])

    def _ours(tiles):
        if ctx['d_to_opp'] is None:
            return list(tiles)
        return [t for t in tiles
                if int(ctx['d_from_me'][t]) < int(ctx['d_to_opp'][t])]

    ctx['d_to_wincoin'] = multi_source_bfs(_ours(game_state['coins']), ctx['blocked'])
    ctx['d_wincoin_here'] = int(ctx['d_to_wincoin'][pos])
    ctx['d_to_mycrate'] = multi_source_bfs(_ours(worth_bombing), ctx['blocked'])
    ctx['d_mycrate_here'] = int(ctx['d_to_mycrate'][pos])

    ctx['d_safe_here'] = int(ctx['now']['d_safe'][pos])
    ctx['in_danger'] = bool(ctx['now']['lethal'][:, pos[0], pos[1]].any())
    return ctx


def afterstate(pos, action, blocked, bombs_left):
# Where we stand after the action and whether if the action was legal.
    if action in MOVES:
        dx, dy = MOVES[action]
        v = (pos[0] + dx, pos[1] + dy)
        return (pos, False) if blocked[v] else (v, True)
    if action == 'BOMB':
        return pos, bombs_left
    return pos, True


def _decay(d):
    return GAMMA_FEAT ** min(int(d), 40) if d != UNREACHABLE else 0.0


def _sign(after, here):
    if after == UNREACHABLE or here == UNREACHABLE:
        return 0.0
    return float(np.sign(after - here))


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

    view = ctx['after_bomb'] if (is_bomb and ctx['bombs_left']) else ctx['now']

    d_here = ctx['d_coin_here']
    if d_here == UNREACHABLE:
        phi[NO_COIN] = 1.0
    else:
        phi[COIN_DELTA] = _sign(int(ctx['d_to_coin'][dest]), d_here)

    dw_here = ctx['d_wincoin_here']
    if dw_here == UNREACHABLE:
        phi[NO_WINCOIN] = 1.0
    else:
        phi[WINCOIN_DELTA] = _sign(int(ctx['d_to_wincoin'][dest]), dw_here)

    dm_here = ctx['d_mycrate_here']
    if dm_here != UNREACHABLE:
        phi[MYCRATE_DELTA] = _sign(int(ctx['d_to_mycrate'][dest]), dm_here)

    dc_here = ctx['d_crate_here']
    if dc_here != UNREACHABLE:
        phi[CRATE_DELTA] = _sign(int(ctx['d_to_crate'][dest]), dc_here)

    if is_bomb and ctx['bombs_left']:
        k = int(ctx['bomb_value'][pos])
        if k == 1:
            phi[CRATES_1] = 1.0
        elif k == 2:
            phi[CRATES_2] = 1.0
        elif k >= 3:
            phi[CRATES_3P] = 1.0

    lethal_at = view['lethal'][:, dest[0], dest[1]]
    if lethal_at.any():
        first = int(np.argmax(lethal_at))
        if first <= 4:
            phi[DANGER_0 + first] = 1.0

    phi[SAFETY_DELTA] = _sign(int(view['d_safe'][dest]), ctx['d_safe_here'])
    phi[NO_ESCAPE] = float(not view['surv'][0][dest])

    free = ctx['field'] == 0
    exits = sum(1 for dx, dy in NEIGHBOURS if free[dest[0] + dx, dest[1] + dy])
    phi[EXITS] = exits / 4.0
    phi[DEAD_END] = float(exits <= 1)

    if ctx['d_to_opp'] is None or ctx['d_opp_here'] == UNREACHABLE:
        phi[NO_OPP] = 1.0
    else:
        phi[OPP_DELTA] = _sign(int(ctx['d_to_opp'][dest]), ctx['d_opp_here'])
        if is_bomb and ctx['bombs_left']:
            hit = set(blast_coords(ctx['field'], pos[0], pos[1]))
            phi[OPP_IN_BLAST] = float(any(o in hit for o in ctx['others']))
            surv = ctx['after_bomb']['surv'][0]
            phi[OPP_TRAPPED] = float(any(o in hit and not surv[o]
                                         for o in ctx['others']))

    phi[X_DANGER_SAFETY] = float(ctx['in_danger']) * phi[SAFETY_DELTA]
    phi[X_BOMB_NOESCAPE] = phi[IS_BOMB] * phi[NO_ESCAPE]
    phi[X_BOMB_CRATES2] = phi[IS_BOMB] * (phi[CRATES_2] + phi[CRATES_3P])
    phi[X_BOMB_OPP] = phi[IS_BOMB] * phi[OPP_IN_BLAST]
    phi[X_BOMB_TRAPPED] = phi[IS_BOMB] * phi[OPP_TRAPPED]

    phi[PHI_STATE] = potential(game_state, ctx)
    return phi


def feature_matrix(game_state, ctx=None):
    if ctx is None:
        ctx = context(game_state)
    return np.stack([features(game_state, a, ctx) for a in ACTIONS])


def potential(game_state, ctx=None):
    #Phi(s) are for potential-based shaping function of the STATE only.
    if game_state is None:
        return 0.0
    if ctx is None:
        ctx = context(game_state)

    phi = 0.25 * _decay(ctx['d_safe_here'])
    if ctx['d_coin_here'] != UNREACHABLE:
        phi += 0.30 * _decay(ctx['d_coin_here'])
    elif ctx['d_crate_here'] != UNREACHABLE:
        phi += W_CRATE * _decay(ctx['d_crate_here'])
    if ctx['bombs_left'] and ctx['d_opp_here'] != UNREACHABLE:
        phi += W_HUNT * _decay(ctx['d_opp_here'])
    return phi


def describe(w, top=None):
    pairs = list(zip(FEATURE_NAMES, w))
    if top:
        pairs = sorted(pairs, key=lambda p: -abs(p[1]))[:top]
    return '  '.join(f'{n}={v:+.3f}' for n, v in pairs)