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
# 4 free directions, bomb availability, coin, crate and enemy directions
# adjacent crates and enemies, danger values, escape directions and bomb quality
# all together we have 37 observations

N_OBSERVATIONS = 37
REVERSAL_Q_PENALTY = 0.5

def setup(self):
    """initialize the network and load the saved model if there is one"""

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
    """find the first direction and distance to the nearest reachable target"""

    if not targets:
        return [0, 0, 0, 0], None

    targets = set(targets)
    if start in targets:
        return [0, 0, 0, 0], 0

    bomb_positions = {bomb_position for bomb_position, _ in bombs}
    enemy_positions = {agent[3] for agent in others}

    # keep the same order as ACTIONS and our features
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
    """get the first BFS direction to the nearest target as one hot"""
    direction, _ = bfs_path_to_nearest_target(
        field,
        start,
        targets,
        bombs,
        others,
    )
    return direction


def bfs_distance_to_nearest_target(field, start, targets, bombs, others):
    """get the BFS distance to the nearest target, None means not reachable"""
    _, distance = bfs_path_to_nearest_target(
        field,
        start,
        targets,
        bombs,
        others,
    )
    return distance


def approach_tiles(field, object_positions, bombs, others):
    """get the free tiles beside crates or opponents"""
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
    """return all tiles hit by the bomb considering walls and crates"""
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
    """get the earliest explosion time for every tile"""
    # -1 means safe, 0 means exploding now and larger values show the time left

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

    # an active explosion can trigger another bomb immediately
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
    """normalize the danger time between 0 and 1"""
    x, y = position

    if x < 0 or y < 0 or x >= danger_map.shape[0] or y >= danger_map.shape[1]:
        return 0.0

    explosion_time = int(danger_map[x, y])

    if explosion_time < 0:
        return 0.0

    return 1.0 / (explosion_time + 1.0)


def evaluate_bomb_placement(game_state):
    """check if planting a bomb is useful and if the agent can escape"""

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
        # planting the bomb takes one step so escaping starts after that
        start_time=1,
        avoid_opponent_interception=True,
    )

    return {
        "escape_possible": escape_possible,
        "crates_hit": int(crates_hit),
        "enemy_in_blast": bool(proposed_blast & enemy_positions),
    }


def get_valid_action_mask(game_state):
    """mask actions that do not have a safe continuation

    I added this because sometimes a move looked safe for one step but the
    agent would get trapped after that
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

        # after moving one step the tile must still be safe
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

    # also check that there is an escape after the first move
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
            # one step to enter the tile plus the rest of the escape path
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
    # if we are already in danger, escape first and do not plant a bomb
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

    # while in danger only keep the fastest escape moves
    # otherwise the agent may change direction and waste the bomb timer
    if current_explosion_time >= 0:
        finite_distances = escape_distances[np.isfinite(escape_distances)]
        if finite_distances.size:
            fastest_escape = finite_distances.min()
            mask &= escape_distances == fastest_escape

    # if no full escape is found, at least use an immediately safe action
    if not mask.any():
        mask = immediate_mask

    # the environment still needs an action even if the agent is trapped
    if not mask.any():
        mask[wait_index] = True

    return mask
    

def act(self, game_state: dict) -> str:
    """choose an action using epsilon greedy and the valid action mask"""
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
        # also during exploration choose only valid actions
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

            # reduce simple A to B to A loops but do not completely block going back
            # we also do not use this when in danger or when there is a coin there
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
        # temporary prints for checking the agent during evaluation
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
    """convert the game state to the 37 features used by the DQN"""
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

    # BFS direction to the nearest coin

    coins = game_state["coins"]

    coin_direction = bfs_direction_to_nearest_target(
        field=field,
        start=(x, y),
        targets=coins,
        bombs=game_state["bombs"],
        others=game_state["others"]
    )
    features.extend(coin_direction)

    # go toward a free tile beside the nearest crate because we cannot walk on it
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

    # go toward a reachable tile beside an opponent
    # if it is already beside us the other features handle the attack decision
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
    """check if a position is now or later inside a blast"""
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
    """get the earliest time an opponent can reach each tile"""
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
    """get the shortest safe escape distance

    None means that no escape was found before max_depth
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

    # check if the agent can walk on a tile
    def is_walkable(position):
        px, py = position

        # do not move outside the board
        if (px < 0 or py < 0 or px >= field.shape[0] or py >= field.shape[1]):
            return False

        # walls and crates are blocked
        if field[px, py] != 0:
            return False

        if position in enemy_positions:
            return False

        # existing bombs are blocked except the bomb below the starting agent
        if position in bomb_positions and position != start:
            return False

        return True

    # there is no escape if the starting tile itself is blocked
    if not is_walkable(start):
        return None

    start_x, start_y = start
    start_danger_time = int(danger_map[start_x, start_y])

    # this tile explodes before or when the agent arrives
    if 0 <= start_danger_time <= start_time:
        return None

    # -1 means this tile is already a safe destination
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

            # an opponent may enter the escape tile at the same time and block us
            # so reject paths where the opponent can arrive first or together
            if (
                enemy_arrival is not None
                and int(enemy_arrival[nx, ny]) <= next_arrival_time
            ):
                continue

            # a dangerous tile can only be crossed before its explosion time
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
    """check if a safe escape exists before max_depth"""
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
