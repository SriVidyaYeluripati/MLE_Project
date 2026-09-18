import math
import os
import random
from collections import deque

import torch

import numpy as np

from .model import DQN

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = "my-saved-model.pt"
# first features count: 4 coins directions, 4 free up, right, down, left, 1 bomb availability
# 4 adjacent crate indicators, 4 adjacents enemy indicators, 4 danger indicators, 1 current danger indicator
# 29 previous features + 4 crate-goal directions + 4 enemy-goal directions
N_OBSERVATIONS = 37
REVERSAL_Q_PENALTY = 0.5

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.policy_net = DQN(n_observations=N_OBSERVATIONS, n_actions=len(ACTIONS)).to(device)

    self.steps_done = 0
    self.position_history = deque(maxlen=4)
    self.position_history_round = None
    self.eps_start = 0.9
    self.eps_end = 0.05
    self.eps_decay = 2500

    if os.path.isfile(MODEL_FILE):
        try:
            self.logger.info("Loading model from saved state.")
            self.policy_net.load_state_dict(_load_model_state(MODEL_FILE))
        except Exception as exc:
            self.logger.warning(f"Could not load saved model, initializing new model: {exc}")
    else:
        self.logger.info("Initializing new model.")

    if self.train:
        self.policy_net.train()
    else:
        self.policy_net.eval()


def _load_model_state(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def bfs_path_to_nearest_target(field, start, targets, bombs, others):
    """Return the first BFS direction and distance to a reachable target."""

    if not targets:
        return [0, 0, 0, 0], None

    targets = set(targets)
    if start in targets:
        return [0, 0, 0, 0], 0

    bomb_positions = {bomb_position for bomb_position, _ in bombs}
    enemy_positions = {agent[3] for agent in others}

    # this order is consistent with the feature vector and ACTIONS
    directions = [
        ((0, -1), [1, 0, 0, 0]),  # UP
        ((1, 0), [0, 1, 0, 0]),   # RIGHT
        ((0, 1), [0, 0, 1, 0]),  # DOWN
        ((-1, 0), [0, 0, 0, 1]), # LEFT
    ]

    def is_walkable(position):
        px, py = position

        if px < 0 or py < 0 or px >= field.shape[0] or py >= field.shape[1]:
            return False

        if field[px, py] != 0:
            return False

        if position in enemy_positions:
            return False

        if position in bomb_positions and position != start:
            return False

        return True

    queue = deque([(start, None, 0)])
    visited = {start}

    while queue:
        current, first_direction, distance = queue.popleft()

        if current in targets:
            return first_direction, distance

        cx, cy = current

        for (dx, dy), direction_one_hot in directions:
            next_position = (cx + dx, cy + dy)

            if next_position in visited or not is_walkable(next_position):
                continue

            visited.add(next_position)

            queue.append(
                (next_position,
                 direction_one_hot if first_direction is None else first_direction,
                 distance + 1,)
            )

    return [0, 0, 0, 0], None


def bfs_direction_to_nearest_target(field, start, targets, bombs, others):
    """Return a one-hot first step toward the nearest reachable target."""
    direction, _ = bfs_path_to_nearest_target(
        field,
        start,
        targets,
        bombs,
        others,
    )
    return direction


def bfs_distance_to_nearest_target(field, start, targets, bombs, others):
    """Return BFS path length to the nearest target, or None if unreachable."""
    _, distance = bfs_path_to_nearest_target(
        field,
        start,
        targets,
        bombs,
        others,
    )
    return distance


def approach_tiles(field, object_positions, bombs, others):
    """Return free tiles from which a crate or opponent can be approached."""
    bomb_positions = {bomb_position for bomb_position, _ in bombs}
    enemy_positions = {agent[3] for agent in others}
    targets = set()

    for object_x, object_y in object_positions:
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            x = int(object_x) + dx
            y = int(object_y) + dy

            if x < 0 or y < 0 or x >= field.shape[0] or y >= field.shape[1]:
                continue

            position = (x, y)
            if (
                field[x, y] == 0
                and position not in bomb_positions
                and position not in enemy_positions
            ):
                targets.add(position)

    return targets

def blast_coordinates(field, bomb_position, blast_range=3):
    """return every tile hit by bombs blast with respect to walls and crates"""
    bomb_x, bomb_y = bomb_position
    blast_tiles = [(bomb_x, bomb_y)]

    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        for distance in range(1, blast_range + 1):
            x = bomb_x + dx * distance
            y = bomb_y + dy * distance

            if x < 0 or y < 0 or x >= field.shape[0] or y >= field.shape[1]:
                break

            if field[x, y] == -1:
                break

            blast_tiles.append((x, y))

            if field[x, y] == 1:
                break

    return blast_tiles

def build_danger_map(field, explosion_map, bombs):
    """return the earliest explosion time for every tile on the game"""
    # -1: no explosion that we know will hit the tile
    # 0: an explosion is already happening on this tile
    # 1: the tile will be explode after the agent's next move
    # 2 and above: the number steps until the explosion happens

    danger_map = np.full(field.shape, -1, dtype=np.int16)
    danger_map[explosion_map > 0] = 0

    if not bombs:
        return danger_map

    explosion_times = {
        bomb_position: int(timer) + 1
        for bomb_position, timer in bombs
    }

    blast_tiles_by_bomb = {
        bomb_position: set(blast_coordinates(field, bomb_position))
        for bomb_position, _ in bombs
    }

    # active explosion can trigger bomb immediately
    for bomb_position in explosion_times:
        bx, by = bomb_position
        if explosion_map[bx, by] > 0:
            explosion_times[bomb_position] = 0


    changed = True
    while changed:
        changed = False

        for source_position, source_blast in blast_tiles_by_bomb.items():
            source_time = explosion_times[source_position]

            for target_position in explosion_times:
                if target_position == source_position:
                    continue

                if (
                    target_position in source_blast
                    and source_time < explosion_times[target_position]
                ):
                    explosion_times[target_position] = source_time
                    changed = True

    for bomb_position, explosion_time in explosion_times.items():
        for x, y in blast_tiles_by_bomb[bomb_position]:
            old_time = danger_map[x, y]

            if old_time == -1 or explosion_time < old_time:
                danger_map[x, y] = explosion_time

    return danger_map

def danger_map_feature(danger_map, position):
    """convert a danger map time to a normalized feature in [0, 1]"""
    x, y = position

    if x < 0 or y < 0 or x >= danger_map.shape[0] or y >= danger_map.shape[1]:
        return 0.0

    explosion_time = int(danger_map[x, y])

    if explosion_time < 0:
        return 0.0

    return 1.0 / (explosion_time + 1.0)


def evaluate_bomb_placement(game_state):
    """ return values are observations for qualityh of planting a bomb and are observations for DQN,
    not action decision"""

    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    _, _, bomb_available, position = game_state["self"]

    if not bomb_available:
        return {
            "escape_possible": False, 
            "crates_hit": 0,
            "enemy_in_blast": False,
        }

    proposed_blast = set(blast_coordinates(field, position))
    enemy_positions = {agent[3] for agent in others}
    crates_hit = sum(field[x, y] == 1 for x, y in proposed_blast)

    hypothetical_bombs = list(bombs)
    hypothetical_bombs.append((position, 3))

    escape_possible = can_escape_from(
        start=position,
        field=field,
        bombs=hypothetical_bombs,
        others=others,
        explosion_map=explosion_map,
        max_depth=6,
        # Planting the bomb consumes the current action. The first escape move
        # therefore reaches a neighboring tile at time 2.
        start_time=1,
        avoid_opponent_interception=True,
    )

    return {
        "escape_possible": escape_possible,
        "crates_hit": int(crates_hit),
        "enemy_in_blast": bool(proposed_blast & enemy_positions),
    }


def get_valid_action_mask(game_state):
    """Return a Boolean mask for actions with a survivable continuation.

    Movement and WAIT must be immediately safe and must leave a time-valid path
    to a tile outside every known blast. BOMB is allowed only when the
    hypothetical bomb still leaves such an escape path. If no fully survivable
    action exists, fall back to immediately safe actions so the framework can
    still receive a legal choice.
    """
    if game_state is None:
        return np.array([False, False, False, False, True, False], dtype=np.bool_)

    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    _, _, bomb_available, (x, y) = game_state["self"]

    bomb_positions = {position for position, _ in bombs}
    enemy_positions = {agent[3] for agent in others}
    danger_map = build_danger_map(field, explosion_map, bombs)

    def movement_is_immediately_safe(position):
        px, py = position

        if px < 0 or py < 0 or px >= field.shape[0] or py >= field.shape[1]:
            return False
        if field[px, py] != 0:
            return False
        if position in bomb_positions or position in enemy_positions:
            return False

        # The movement arrives at time 1. Do not enter a tile that explodes
        # before or exactly when the agent gets there.
        explosion_time = int(danger_map[px, py])
        return explosion_time < 0 or explosion_time > 1

    movement_positions = [
        (x, y - 1),  # UP
        (x + 1, y),  # RIGHT
        (x, y + 1),  # DOWN
        (x - 1, y),  # LEFT
    ]

    immediate_mask = np.array(
        [
            movement_is_immediately_safe(position)
            for position in movement_positions
        ] + [
            False,  # WAIT, filled below
            False,  # BOMB, filled below
        ],
        dtype=np.bool_,
    )

    current_explosion_time = int(danger_map[x, y])
    immediate_mask[ACTIONS.index("WAIT")] = (
        current_explosion_time < 0 or current_explosion_time > 1
    )

    # Validate the complete continuation, not only the next tile. This prevents
    # choosing a direction that looks safe for one step but ends in a dead end
    # before a known explosion.
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)

    escape_distances = np.full(len(ACTIONS), np.inf, dtype=np.float32)

    for action_index, position in enumerate(movement_positions):
        if not immediate_mask[action_index]:
            continue

        distance = escape_distance_from(
            start=position,
            field=field,
            bombs=bombs,
            others=others,
            explosion_map=explosion_map,
            max_depth=6,
            danger_map=danger_map,
            start_time=1,
        )
        if distance is not None:
            mask[action_index] = True
            # One action is needed to enter `position`, followed by `distance`
            # additional moves to reach a tile outside all known blasts.
            escape_distances[action_index] = 1 + distance

    wait_index = ACTIONS.index("WAIT")
    if immediate_mask[wait_index]:
        distance = escape_distance_from(
            start=(x, y),
            field=field,
            bombs=bombs,
            others=others,
            explosion_map=explosion_map,
            max_depth=6,
            danger_map=danger_map,
            start_time=1,
        )
        if distance is not None:
            mask[wait_index] = True
            escape_distances[wait_index] = 1 + distance

    bomb_index = ACTIONS.index("BOMB")
    # Do not spend another step planting a bomb while already inside a known
    # blast path. Escape first; bombing decisions resume from a safe tile.
    if bomb_available and current_explosion_time < 0:

        bomb_quality = evaluate_bomb_placement(game_state)
        bomb_is_useful = (
            bomb_quality["crates_hit"] > 0
            or bomb_quality["enemy_in_blast"]
            )

        bomb_is_valid = (
            bomb_quality["escape_possible"]
            and bomb_is_useful
        )
        immediate_mask[bomb_index] = bomb_is_valid
        mask[bomb_index] = bomb_is_valid

    # When danger is already present, merely knowing that several actions have
    # an escape continuation is not enough: repeatedly selecting different
    # "still escapable" actions can waste the remaining bomb timer. Keep only
    # first steps belonging to a fastest known escape. Outside danger, the DQN
    # retains the full valid-action choice for collecting, bombing and combat.
    if current_explosion_time >= 0:
        finite_distances = escape_distances[np.isfinite(escape_distances)]
        if finite_distances.size:
            fastest_escape = finite_distances.min()
            mask &= escape_distances == fastest_escape

    # Known blast paths can occasionally leave no provably survivable choice.
    # Prefer an immediately safe legal action over forcing WAIT in that case.
    if not mask.any():
        mask = immediate_mask

    # If the agent is already trapped, the framework still requires an action.
    if not mask.any():
        mask[wait_index] = True

    return mask
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)


    current_position = game_state["self"][3]
    if self.position_history_round != game_state["round"]:
        self.position_history.clear()
        self.position_history_round = game_state["round"]

    previous_position = (
        self.position_history[-1]
        if self.position_history
        else None
    )

    features = state_to_features(game_state)

    # this line is just for debugging we want to see the alterng of features and how they look like
    #print("Features:", features.cpu().numpy())

    eps_threshold = (
        self.eps_end + (self.eps_start - self.eps_end) *
        math.exp(-self.steps_done / self.eps_decay)
    )

    self.steps_done += 1

    if features is None:
        return "WAIT"

    valid_action_mask = get_valid_action_mask(game_state)
    valid_action_indices = np.flatnonzero(valid_action_mask).tolist()

    if self.train and random.random() < eps_threshold:
        # Exploration must also choose only valid actions.
        action = random.choice(valid_action_indices)
    else:
        with torch.no_grad():
            q_values = self.policy_net(features)
            mask_tensor = torch.tensor(
                valid_action_mask,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            q_values = q_values.masked_fill(~mask_tensor, float("-inf"))

            # Break deterministic A->B->A oscillations softly. This is not a
            # hard mask: returning remains possible when it is genuinely much
            # better. Never apply it during bomb danger or when returning would
            # immediately collect a visible coin.
            danger_map_for_choice = build_danger_map(
                game_state["field"],
                game_state["explosion_map"],
                game_state["bombs"],
            )
            x, y = current_position
            if (
                previous_position is not None
                and int(danger_map_for_choice[x, y]) < 0
                and previous_position not in game_state["coins"]
            ):
                next_positions = [
                    (x, y - 1),
                    (x + 1, y),
                    (x, y + 1),
                    (x - 1, y),
                ]
                for action_index, position in enumerate(next_positions):
                    if position == previous_position and valid_action_mask[action_index]:
                        q_values[0, action_index] -= REVERSAL_Q_PENALTY

        action = q_values.argmax(dim=1).item()
        # Temporary evaluation debugging
        if not self.train and os.environ.get("DQN_DEBUG") == "1":
            x, y = game_state["self"][3]

            danger_map = build_danger_map(
                game_state["field"],
                game_state["explosion_map"],
                game_state["bombs"],
            )

            bomb_quality = evaluate_bomb_placement(game_state)

            print(
                "\nRound/step:",
                (game_state["round"], game_state["step"]),
            )
            print("Position:", (x, y))
            print("Coins:", game_state["coins"])
            print(
                "Opponents:",
                [agent[3] for agent in game_state["others"]],
            )
            print("Bombs:", game_state["bombs"])
            print("Bomb quality:", bomb_quality)
            print(
                "Current danger time:",
                int(danger_map[x, y])
            )
            print(
                "Valid:",
                dict(zip(ACTIONS, valid_action_mask.tolist()))
            )
            print(
                "Masked Q-values:",
                dict(
                    zip(
                        ACTIONS,
                        q_values.squeeze(0).cpu().tolist()
                    )
                )
            )
            print("Chosen:", ACTIONS[action])
    self.position_history.append(current_position)
    return ACTIONS[action]


def state_to_features(game_state: dict) -> torch.Tensor:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    field = game_state["field"]

    explosion_map = game_state["explosion_map"]

    bombs = game_state["bombs"]
    danger_map = build_danger_map(field, explosion_map, bombs)

    bomb_positions = {bomb_pos for bomb_pos, timer in bombs}
    enemy_positions = {agent[3] for agent in game_state["others"]}

    _, _, bomb_available, (x, y) = game_state["self"]

    features = []

    def is_free(position):
        px, py = position

        return (
            field[px, py] == 0 and
            position not in bomb_positions and
            position not in enemy_positions
        )

    # adjacent tiles
    features.append(int(is_free((x, y - 1))))  # UP
    features.append(int(is_free((x + 1, y))))  # RIGHT
    features.append(int(is_free((x, y + 1))))  # DOWN
    features.append(int(is_free((x - 1, y))))  # LEFT

    # bomb availability
    features.append(int(bomb_available))

    # This is after first failure
    # directions of nearest coin

    coins = game_state["coins"]

    coin_direction = bfs_direction_to_nearest_target(
        field=field,
        start=(x, y),
        targets=coins,
        bombs=game_state["bombs"],
        others=game_state["others"]
    )
    features.extend(coin_direction)

    # BFS direction to the nearest reachable tile beside a crate. Crates are
    # not walkable targets themselves, so their free neighboring tiles are the
    # useful bomb-placement goals.
    crate_positions = [
        tuple(position)
        for position in np.argwhere(field == 1)
    ]
    crate_approach_tiles = approach_tiles(
        field,
        crate_positions,
        bombs,
        game_state["others"],
    )
    crate_direction = bfs_direction_to_nearest_target(
        field=field,
        start=(x, y),
        targets=crate_approach_tiles,
        bombs=bombs,
        others=game_state["others"],
    )
    features.extend(crate_direction)

    # BFS direction to a reachable tile beside an opponent. When already next
    # to an opponent this direction is all zero, while the existing adjacent
    # enemy and bomb-quality features describe the immediate attack decision.
    enemy_approach_tiles = approach_tiles(
        field,
        enemy_positions,
        bombs,
        game_state["others"],
    )
    enemy_direction = bfs_direction_to_nearest_target(
        field=field,
        start=(x, y),
        targets=enemy_approach_tiles,
        bombs=bombs,
        others=game_state["others"],
    )
    features.extend(enemy_direction)


    # crate indicators
    features.append(int(field[x, y - 1] == 1)) # up
    features.append(int(field[x + 1, y] == 1)) # right
    features.append(int(field[x, y + 1] == 1)) # down
    features.append(int(field[x - 1, y] == 1)) # left

    # enemy indicators
    other_agents = game_state["others"]
    features.append(int(any(agent[3] == (x, y - 1) for agent in other_agents))) # up
    features.append(int(any(agent[3] == (x + 1, y) for agent in other_agents))) # right
    features.append(int(any(agent[3] == (x, y + 1) for agent in other_agents))) # down
    features.append(int(any(agent[3] == (x - 1, y) for agent in other_agents))) # left

    # time aware danger: 0 means safe; larger values mean sooner.
    features.append(danger_map_feature(danger_map, (x, y - 1))) # up
    features.append(danger_map_feature(danger_map, (x + 1, y))) # right
    features.append(danger_map_feature(danger_map, (x, y + 1))) # down
    features.append(danger_map_feature(danger_map, (x -1, y))) # left

    # current tile danger
    features.append(danger_map_feature(danger_map, (x, y)))


    # escape
    escape_up = can_escape_from(
        (x, y - 1),
        field,
        bombs,
        other_agents,
        explosion_map,
        danger_map=danger_map,
        start_time=1,
    )

    
    escape_right = can_escape_from(
        (x + 1, y),
        field,
        bombs,
        other_agents,
        explosion_map,
        danger_map=danger_map,
        start_time=1,
    )

    escape_down = can_escape_from(
        (x, y + 1),
        field,
        bombs,
        other_agents,
        explosion_map,
        danger_map=danger_map,
        start_time=1,
    )

    escape_left = can_escape_from(
        (x - 1, y),
        field,
        bombs,
        other_agents,
        explosion_map,
        danger_map=danger_map,
        start_time=1,
    )


    features.append(int(escape_up))
    features.append(int(escape_right))
    features.append(int(escape_down))
    features.append(int(escape_left))

    # bomb placement evaluation
    bomb_quality = evaluate_bomb_placement(game_state)
    features.append(float(bomb_quality["escape_possible"]))
    features.append(bomb_quality["crates_hit"] / 4.0)
    features.append(float(bomb_quality["enemy_in_blast"]))


    features = np.asarray(features, dtype=np.float32)

    return torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


def is_position_danger(field, explosion_map, position, bombs):
    """
    Return True if the position is currently or eventually affected
    by an explosion.
    """
    x, y = position

    if (
        x < 0
        or y < 0
        or x >= field.shape[0]
        or y >= field.shape[1]
    ):
        return False

    danger_map = build_danger_map(
        field,
        explosion_map,
        bombs,
    )

    return danger_map[x, y] >= 0


def opponent_arrival_map(field, bombs, others, max_depth=6):
    """Earliest number of moves in which an opponent can reach each tile."""
    distances = np.full(field.shape, max_depth + 1, dtype=np.int16)
    bomb_positions = {position for position, _ in bombs}
    queue = deque()

    for agent in others:
        position = agent[3]
        distances[position] = 0
        queue.append(position)

    while queue:
        x, y = queue.popleft()
        distance = int(distances[x, y])
        if distance >= max_depth:
            continue

        for position in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            px, py = position
            if px < 0 or py < 0 or px >= field.shape[0] or py >= field.shape[1]:
                continue
            if field[px, py] != 0 or position in bomb_positions:
                continue
            if distances[px, py] <= distance + 1:
                continue
            distances[px, py] = distance + 1
            queue.append(position)

    return distances


def escape_distance_from(start,
    field,
    bombs,
    others,
    explosion_map,
    max_depth=6,
    danger_map=None,
    start_time=0,
    avoid_opponent_interception=False,):
    """Return the shortest time-valid distance to a tile outside known blasts.

    ``None`` means that no escape was found within ``max_depth``. ``start_time``
    is the time at which the agent reaches ``start`` relative to the current
    game state.
    """
    enemy_positions = {agent[3] for agent in others}

    bomb_positions = {bomb_pos for bomb_pos, timer in bombs}

    if danger_map is None:
        danger_map = build_danger_map(field, explosion_map, bombs)

    enemy_arrival = None
    if avoid_opponent_interception and others:
        enemy_arrival = opponent_arrival_map(
            field,
            bombs,
            others,
            max_depth=max_depth + start_time,
        )

    # walkable
    def is_walkable(position):
        px, py = position

        # prevent moving outside the board
        if (px < 0 or py < 0 or px >= field.shape[0] or py >= field.shape[1]):
            return False

        # walls and crates cant be walked on
        if field[px, py] != 0:
            return False

        if position in enemy_positions:
            return False

        # existing bombs cant be walked on
        if position in bomb_positions and position != start:
            return False

        return True

    # We cannot escape through this direction if the starting tile is blocked.
    if not is_walkable(start):
        return None

    start_x, start_y = start
    start_danger_time = int(danger_map[start_x, start_y])

    # The explosion happens no later than our arrival on this tile.
    if 0 <= start_danger_time <= start_time:
        return None

    # A tile that is absent from the danger map is a valid safe destination.
    if start_danger_time == -1:
        return 0

    queue = deque([(start, start_time)])

    visited = {start}

    while queue:
        current_position, arrival_time = queue.popleft()

        if arrival_time - start_time >= max_depth:
            continue

        cx, cy = current_position

        neighbors = [
            (cx, cy -1),
            (cx + 1, cy),
            (cx, cy + 1),
            (cx - 1, cy),
        ]

        for next_position in neighbors:

            if next_position in visited:
                continue

            if not is_walkable(next_position):
                continue

            nx, ny = next_position
            next_arrival_time = arrival_time + 1
            next_danger_time = int(danger_map[nx, ny])

            # In simultaneous play an opponent can enter our intended escape
            # tile in the same step. The engine then rejects our movement and
            # leaves us inside the blast. A bomb is considered safely
            # escapable only if the planned path cannot be intercepted this
            # quickly by a currently visible opponent.
            if (
                enemy_arrival is not None
                and int(enemy_arrival[nx, ny]) <= next_arrival_time
            ):
                continue

            # It is safe to cross a future blast tile only before it explodes.
            if 0 <= next_danger_time <= next_arrival_time:
                continue

            if next_danger_time == -1:
                return next_arrival_time - start_time

            visited.add(next_position)
            queue.append((next_position, next_arrival_time))

    return None


def can_escape_from(start,
    field,
    bombs,
    others,
    explosion_map,
    max_depth=6,
    danger_map=None,
    start_time=0,
    avoid_opponent_interception=False,):
    """Return whether a time-valid escape exists within ``max_depth``."""
    return escape_distance_from(
        start=start,
        field=field,
        bombs=bombs,
        others=others,
        explosion_map=explosion_map,
        max_depth=max_depth,
        danger_map=danger_map,
        start_time=start_time,
        avoid_opponent_interception=avoid_opponent_interception,
    ) is not None
