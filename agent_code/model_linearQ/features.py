"""
Feature extraction.   Q(s, a) = w . phi(s, a)

Features are ACTION-RELATIVE and share one weight vector: phi describes what an
action *does*, not where we are.  That is why "move toward the nearest coin" is
learned once instead of four times, and why nothing here refers to absolute
board coordinates.

Stage 1 of the feature roll-out (task 1: collect revealed coins).
Indices are stable - later stages append, they never renumber.
"""

from collections import deque

import numpy as np

try:                                  # loaded as a package by the framework
    from .danger import precompute, MOVES
except ImportError:                   # run directly, e.g. by the tests
    from danger import precompute, MOVES

# Action set. BOMB joins at stage 2; keeping it out now means the agent cannot
# waste steps or kill itself while we are only validating the pipeline.
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

FEATURE_NAMES = [
    'bias',            # 0  constant
    'is_wait',         # 1  this action is WAIT
    'is_invalid',      # 2  blocked by wall, crate, bomb or agent
    'd_coin',          # 3  gamma^(steps to nearest reachable coin), afterstate
    'coin_delta',      # 4  -1 closer, 0 same, +1 further away
    'no_coin',         # 5  no coin currently reachable
]
N_FEATURES = len(FEATURE_NAMES)
BIAS, IS_WAIT, IS_INVALID, D_COIN, COIN_DELTA, NO_COIN = range(N_FEATURES)

GAMMA_FEAT = 0.9      # decay used inside distance features (not the learner's gamma)
UNREACHABLE = np.iinfo(np.int32).max


def multi_source_bfs(sources, blocked):
    """
    Distance from every tile to the nearest source, ignoring time.
    One call per step gives the coin distance for all six afterstates at once.
    """
    dist = np.full(blocked.shape, UNREACHABLE, dtype=np.int32)
    q = deque()
    for s in sources:
        if not blocked[s]:
            dist[s] = 0
            q.append(s)
    while q:
        x, y = q.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            v = (x + dx, y + dy)
            if not (0 <= v[0] < blocked.shape[0] and 0 <= v[1] < blocked.shape[1]):
                continue
            if blocked[v] or dist[v] != UNREACHABLE:
                continue
            dist[v] = dist[(x, y)] + 1
            q.append(v)
    return dist


def context(game_state):
    """Everything computed once per step and shared by all actions."""
    ctx = precompute(game_state)
    # Coins are collected by walking onto them, so they are not obstacles.
    ctx['d_to_coin'] = multi_source_bfs(game_state['coins'], ctx['blocked'])
    ctx['d_coin_here'] = int(ctx['d_to_coin'][ctx['pos']])
    return ctx


def afterstate(pos, action, blocked):
    """Where we stand after the action, and whether the action was legal."""
    if action in MOVES:
        dx, dy = MOVES[action]
        v = (pos[0] + dx, pos[1] + dy)
        if blocked[v]:
            return pos, False        # bumping a wall: we do not move
        return v, True
    return pos, True                 # WAIT (and later BOMB) never moves us


def features(game_state, action, ctx=None):
    """phi(s, a) for one action. Returns a float array of shape (N_FEATURES,)."""
    if ctx is None:
        ctx = context(game_state)

    phi = np.zeros(N_FEATURES)
    phi[BIAS] = 1.0
    phi[IS_WAIT] = float(action == 'WAIT')

    pos_after, legal = afterstate(ctx['pos'], action, ctx['blocked'])
    phi[IS_INVALID] = float(not legal)

    d_here = ctx['d_coin_here']
    d_after = int(ctx['d_to_coin'][pos_after])

    if d_here == UNREACHABLE:
        phi[NO_COIN] = 1.0
    else:
        phi[D_COIN] = GAMMA_FEAT ** min(d_after, 40) if d_after != UNREACHABLE else 0.0
        # signed, single feature: "toward" and "away" can never both pay
        phi[COIN_DELTA] = float(np.sign(d_after - d_here)) if d_after != UNREACHABLE else 1.0

    return phi


def feature_matrix(game_state, ctx=None):
    """phi for every action, shape (len(ACTIONS), N_FEATURES)."""
    if ctx is None:
        ctx = context(game_state)
    return np.stack([features(game_state, a, ctx) for a in ACTIONS])


def potential(game_state, ctx=None):
    """
    Phi(s) for potential-based reward shaping. A function of the STATE only -
    no action appears anywhere - so gamma*Phi(s') - Phi(s) cannot change the
    set of optimal policies, and cannot pay for a back-and-forth loop.
    """
    if game_state is None:
        return 0.0                   # Phi(terminal) = 0, always
    if ctx is None:
        ctx = context(game_state)
    d = ctx['d_coin_here']
    return 0.0 if d == UNREACHABLE else 0.3 * (GAMMA_FEAT ** min(d, 40))


def describe(w):
    """Human-readable weight vector - the whole reason this model was chosen."""
    return '  '.join(f'{n}={v:+.3f}' for n, v in zip(FEATURE_NAMES, w))
