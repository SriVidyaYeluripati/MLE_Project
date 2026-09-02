"""
Shared feature extraction: game_state dict -> fixed-length float vector.

============================ TEAM CONTRACT ============================
Model A (this folder) and Model B (the DQN) both import from here so
they run on IDENTICAL input. That makes "Model A vs Model B" a real
controlled comparison instead of two unrelated things.

To use in the DQN, in its callbacks.py:
    from .features import state_to_features, N_FEATURES
    N_OBSERVATIONS = N_FEATURES        # = 28

Anyone who changes FEATURE_NAMES must tell the whole team, because it
changes the input size of every model.
======================================================================
"""
import numpy as np
from collections import deque

# 4 directions in the order UP, RIGHT, DOWN, LEFT  (matches ACTIONS order)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

FEATURE_NAMES = (
    [f"free_{d}"      for d in "URDL"] +   # 4  is the neighbouring tile walkable
    [f"coin_dir_{d}"  for d in "URDL"] +   # 4  one-hot: BFS step toward nearest coin
    [f"crate_dir_{d}" for d in "URDL"] +   # 4  one-hot: BFS step toward nearest crate
    [f"danger_{d}"    for d in "URDL"] +   # 4  danger of neighbour tile in [0,1]
    ["danger_here"] +                      # 1  danger of my own tile
    [f"escape_{d}"    for d in "URDL"] +   # 4  moving there leads to a safe tile
    ["bomb_available"] +                   # 1
    ["bomb_here_is_useful"] +              # 1  bombing now would hit a crate/enemy
    ["escape_if_bomb_here"] +              # 1  I could survive my own bomb
    [f"enemy_dir_{d}" for d in "URDL"]     # 4  one-hot: BFS step toward nearest enemy
)
N_FEATURES = len(FEATURE_NAMES)   # 28


def _blast_coords(x, y, field):
    """Tiles hit by a bomb at (x, y). Stops at stone walls (field == -1)."""
    coords = [(x, y)]
    for dx, dy in DIRS:
        for i in range(1, 4):          # BOMB_POWER = 3
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:    # stone wall blocks the blast
                break
            coords.append((nx, ny))
    return coords


def danger_map(game_state):
    """Per-tile danger in [0, 1]. 1.0 = exploding now / imminent, 0 = safe."""
    field = game_state["field"]
    danger = np.array(game_state["explosion_map"], dtype=float)
    danger = np.where(danger > 0, 1.0, 0.0)          # active explosions
    for (bx, by), timer in game_state["bombs"]:
        # timer counts down 3..0; smaller timer = more urgent
        urgency = (4.0 - timer) / 4.0
        for (cx, cy) in _blast_coords(bx, by, field):
            danger[cx, cy] = max(danger[cx, cy], urgency)
    return danger


def _bfs_first_step(start, field, targets, blocked):
    """
    One-hot over DIRS: the first move of the shortest path to the nearest
    target. Returns 4 zeros if no target is reachable.
    """
    out = np.zeros(4)
    if not targets:
        return out
    targets = set(targets)
    q = deque()                       # holds (position, index of first move)
    seen = {start}
    for i, (dx, dy) in enumerate(DIRS):
        n = (start[0] + dx, start[1] + dy)
        if n in targets:              # target is the adjacent tile itself
            out[i] = 1.0
            return out
        if field[n] == 0 and n not in blocked:
            q.append((n, i))
            seen.add(n)
    while q:
        (cx, cy), first = q.popleft()
        for dx, dy in DIRS:
            n = (cx + dx, cy + dy)
            if n in seen:
                continue
            if n in targets:          # reachable target (may be a crate wall)
                out[first] = 1.0
                return out
            if field[n] == 0 and n not in blocked:
                seen.add(n)
                q.append((n, first))
    return out


def _can_escape(start, field, danger, others, max_depth=5):
    """BFS: is a fully safe tile reachable within max_depth steps?"""
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (cx, cy), d = q.popleft()
        if danger[cx, cy] == 0 and d > 0:
            return True
        if d >= max_depth:
            continue
        for dx, dy in DIRS:
            n = (cx + dx, cy + dy)
            if n in seen or field[n] != 0 or n in others:
                continue
            seen.add(n)
            q.append((n, d + 1))
    return False


def state_to_features(game_state):
    """game_state dict -> np.array of shape (N_FEATURES,), dtype float32."""
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_available, (x, y) = game_state["self"]
    coins = [tuple(c) for c in game_state["coins"]]
    others = [tuple(o[3]) for o in game_state["others"]]
    danger = danger_map(game_state)
    blocked = set(others) | {tuple(b[0]) for b in game_state["bombs"]}

    f = []

    # --- navigation: is each neighbour walkable ---
    free = []
    for dx, dy in DIRS:
        n = (x + dx, y + dy)
        free.append(float(field[n] == 0 and n not in blocked))
    f += free

    # --- objectives: direction to nearest coin / crate ---
    f += list(_bfs_first_step((x, y), field, coins, blocked))
    crates = [tuple(c) for c in np.argwhere(field == 1)]
    f += list(_bfs_first_step((x, y), field, crates, blocked))

    # --- survival: danger of each neighbour, and of my own tile ---
    for dx, dy in DIRS:
        f.append(float(danger[x + dx, y + dy]))
    f.append(float(danger[x, y]))

    # --- escape: does stepping this way lead somewhere safe ---
    for i, (dx, dy) in enumerate(DIRS):
        n = (x + dx, y + dy)
        f.append(1.0 if free[i] and _can_escape(n, field, danger, blocked) else 0.0)

    # --- bombing ---
    f.append(float(bomb_available))
    blast = _blast_coords(x, y, field)
    useful = any(field[c] == 1 for c in blast) or any(o in blast for o in others)
    f.append(float(useful))
    hypo = danger.copy()
    for (cx, cy) in blast:
        hypo[cx, cy] = 1.0
    f.append(float(_can_escape((x, y), field, hypo, blocked)))

    # --- direction to nearest enemy ---
    f += list(_bfs_first_step((x, y), field, others, blocked))

    return np.array(f, dtype=np.float32)
