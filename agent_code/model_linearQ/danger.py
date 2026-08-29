"""
Danger map and time-aware BFS.  Step 1 of the build order in "Building Agent B".

Everything downstream (features, rewards, diagnostics) is built on these two
functions, so they are kept free of any learning code and covered by tests.

Timing facts, verified against the framework (settings.py, items.py,
environment.py) rather than assumed:

  BOMB_POWER      = 3      blast reaches 3 tiles in each direction
  BOMB_TIMER      = 4      countdown seen in game_state['bombs']
  EXPLOSION_TIMER = 2      the blast is lethal on 2 consecutive steps
  get_blast_coords breaks only on arena == -1  ->  CRATES DO NOT SHIELD YOU

Order inside environment.do_step():
    agents act  ->  collect_coins  ->  update_explosions  ->  update_bombs
    ->  evaluate_explosions (this is where agents die)

Let step j = 0 be the step we are deciding right now, so the tile we occupy
after our action is evaluated for lethality at j = 0.

  * a bomb observed with timer t detonates at step j = t, and its tiles are
    lethal at j = t and j = t + 1.
  * explosion_map[x, y] >= 1 means that tile is still lethal at j = 0.
    (A value of 0 is always safe: such an explosion turns to smoke in
    update_explosions, before evaluate_explosions runs.)
"""

import numpy as np

BOMB_POWER = 3
EXPLOSION_TIMER = 2
HORIZON = 7  # max bomb timer 4 -> lethal up to j = 5; 7 gives headroom

MOVES = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}


def blast_coords(field, bx, by, power=BOMB_POWER):
    """Tiles covered by a bomb at (bx, by). Mirrors Bomb.get_blast_coords."""
    coords = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            x, y = bx + i * dx, by + i * dy
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
            if field[x, y] == -1:      # only stone walls stop a blast
                break
            coords.append((x, y))
    return coords


def danger_map(game_state, horizon=HORIZON):
    """
    lethal[j, x, y] is True if standing on (x, y) at the end of step j kills us.
    """
    field = game_state['field']
    lethal = np.zeros((horizon,) + field.shape, dtype=bool)

    # bombs still ticking
    for (bx, by), t in game_state['bombs']:
        for (x, y) in blast_coords(field, bx, by):
            for j in (t, t + 1):
                if j < horizon:
                    lethal[j, x, y] = True

    # blasts already burning: lethal for this step only
    exp = game_state['explosion_map']
    lethal[0][exp >= 1] = True

    return lethal


def blocked_map(game_state):
    """Tiles we cannot move onto: walls, crates, bombs, other agents."""
    field = game_state['field']
    blocked = field != 0
    for (bx, by), _ in game_state['bombs']:
        blocked[bx, by] = True
    for _, _, _, (ox, oy) in game_state['others']:
        blocked[ox, oy] = True
    return blocked


def safe_bfs(start, blocked, lethal, horizon=HORIZON):
    """
    Time-aware breadth-first search over (x, y, j).

    Returns
        reach : list of sets, reach[j] = tiles we can occupy at end of step j
                without having died on the way
        dist  : int array, dist[x, y] = fewest actions to stand safely on
                (x, y); UNREACHABLE where no safe path exists
    """
    UNREACHABLE = np.iinfo(np.int32).max
    dist = np.full(blocked.shape, UNREACHABLE, dtype=np.int32)
    dist[start] = 0

    frontier = {start}
    reach = []
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
            # we are dead no matter what we do; remaining sets stay empty
            reach.extend(set() for _ in range(horizon - 1 - j))
            break

    return reach, dist


def precompute(game_state):
    """One danger map and one BFS per step, reused by all six actions."""
    _, _, _, pos = game_state['self']
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