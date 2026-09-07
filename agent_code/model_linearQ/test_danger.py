"""
Hand-drawn board tests for danger.py.   Run with:  python -m pytest -q  (or plain python)

Board legend, one character per tile, rows are y and columns are x:
    #  stone wall        c  crate          .  free
    A  our agent         O  an opponent    $  coin
    0-4  a bomb with that timer            *  burning blast (explosion_map = 1)
"""

import numpy as np
from danger import danger_map, blocked_map, safe_bfs, precompute, blast_coords


def make_state(rows, step=1):
    """Turn an ASCII board into a game_state dict."""
    rows = [r for r in rows.strip('\n').split('\n')]
    h, w = len(rows), len(rows[0])
    field = np.zeros((w, h), dtype=int)          # field[x, y]
    explosion_map = np.zeros((w, h))
    bombs, coins, others, me = [], [], [], None

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                field[x, y] = -1
            elif ch == 'c':
                field[x, y] = 1
            elif ch == 'A':
                me = (x, y)
            elif ch == 'O':
                others.append(('opp', 0, True, (x, y)))
            elif ch == '$':
                coins.append((x, y))
            elif ch == '*':
                explosion_map[x, y] = 1
            elif ch.isdigit():
                bombs.append(((x, y), int(ch)))

    assert me is not None, 'board needs an A'
    return {
        'round': 1, 'step': step, 'field': field, 'bombs': bombs,
        'explosion_map': explosion_map, 'coins': coins,
        'self': ('me', 0, True, me), 'others': others, 'user_input': None,
    }


# --------------------------------------------------------------------------- #
# blast geometry
# --------------------------------------------------------------------------- #

def test_blast_stops_at_wall_but_not_at_crate():
    st = make_state("""
#######
#A.c..#
#######
""")
    coords = set(blast_coords(st['field'], 1, 1))
    # reaches 3 right, passing straight through the crate at x = 3
    assert (3, 1) in coords, 'crate tile itself is hit'
    assert (4, 1) in coords, 'crates must NOT shield the blast'
    assert (5, 1) not in coords, 'power is 3'
    assert (0, 1) not in coords, 'stone wall stops it'


# --------------------------------------------------------------------------- #
# lethality timing
# --------------------------------------------------------------------------- #

def test_bomb_is_lethal_on_its_own_step_and_the_next():
    st = make_state("""
#####
#A2.#
#####
""")
    lethal = danger_map(st)
    assert not lethal[1][1, 1], 'timer 2 -> safe at j = 1'
    assert lethal[2][1, 1], 'detonates at j = 2'
    assert lethal[3][1, 1], 'still burning at j = 3'
    assert not lethal[4][1, 1], 'smoke by j = 4'


def test_burning_blast_is_lethal_now_only():
    st = make_state("""
#####
#A*.#
#####
""")
    lethal = danger_map(st)
    assert lethal[0][2, 1], 'explosion_map >= 1 kills at the end of this step'
    assert not lethal[1][2, 1]


# --------------------------------------------------------------------------- #
# escaping
# --------------------------------------------------------------------------- #

def test_dead_end_with_own_bomb_has_no_escape():
    # corridor of length 3, bomb just dropped at the closed end, we are on it
    st = make_state("""
######
#A4..#
######
""")
    st['bombs'] = [((1, 1), 4)]
    ctx = precompute(st)
    assert not ctx['escape_exists'], 'blast covers the whole reachable corridor'


def test_long_corridor_allows_outrunning_the_bomb():
    st = make_state("""
##########
#A4......#
##########
""")
    st['bombs'] = [((1, 1), 4)]
    ctx = precompute(st)
    assert ctx['escape_exists'], 'five tiles is enough to outrun a 3-tile blast'
    assert ctx['dist'][5, 1] == 4, 'four moves to reach the first safe tile'


def test_side_pocket_is_a_valid_escape():
    st = make_state("""
#####
#A4.#
##.##
#####
""")
    st['bombs'] = [((1, 1), 4)]
    ctx = precompute(st)
    assert ctx['escape_exists'], 'stepping around the corner escapes the blast'


# --------------------------------------------------------------------------- #
# obstacles
# --------------------------------------------------------------------------- #

def test_bombs_and_opponents_block_movement():
    st = make_state("""
#####
#A2O#
#####
""")
    blocked = blocked_map(st)
    assert blocked[2, 1], 'a bomb blocks'
    assert blocked[3, 1], 'an opponent blocks'


def test_distance_is_measured_around_crates():
    st = make_state("""
#####
#A.$#
#####
""")
    ctx = precompute(st)
    assert ctx['dist'][3, 1] == 2, 'two steps to the coin'


if __name__ == '__main__':
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f'  ok   {name}')
                passed += 1
            except AssertionError as exc:
                print(f'  FAIL {name}: {exc}')
                failed += 1
    print(f'\n{passed} passed, {failed} failed')