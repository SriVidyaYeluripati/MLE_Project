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
N_OBSERVATIONS = 26
MOVE_DELTAS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}

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

    valid_actions = get_valid_actions(game_state)

    if self.train and random.random() < eps_threshold:
        weights = np.array(
            [0.22 if action in MOVE_DELTAS else 0.08 if action == "WAIT" else 0.04
             for action in valid_actions],
            dtype=np.float32
        )
        weights = weights / weights.sum()
        return np.random.choice(valid_actions, p=weights)
    else:
        with torch.no_grad():
            q_values = self.policy_net(features)
            invalid_actions = [
                action_index for action_index, action in enumerate(ACTIONS)
                if action not in valid_actions
            ]
            if invalid_actions:
                q_values[:, invalid_actions] = -float("inf")
            # print("Features:", features.cpu().numpy())
            # print("Q-values:", q_values.cpu().numpy())
            # print("Chosen:", ACTIONS[q_values.argmax(dim=1).item()])
        action = q_values.argmax(dim=1).item()

    return ACTIONS[action]


def get_valid_actions(game_state: dict) -> list:
    if game_state is None:
        return ["WAIT"]

    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    bomb_positions = {bomb_pos for bomb_pos, timer in bombs}
    enemy_positions = {agent[3] for agent in others}
    _, _, bomb_available, (x, y) = game_state["self"]
    current_position = (x, y)
    current_danger = is_position_danger(field, explosion_map, current_position, bombs)

    safe_moves = []
    legal_moves = []
    for action, (dx, dy) in MOVE_DELTAS.items():
        position = (x + dx, y + dy)
        if not _is_free_tile(field, position, bomb_positions, enemy_positions):
            continue

        legal_moves.append(action)
        if not is_position_danger(field, explosion_map, position, bombs):
            safe_moves.append(action)

    valid_actions = safe_moves or legal_moves

    if not current_danger:
        valid_actions.append("WAIT")

    if _can_place_useful_safe_bomb(game_state):
        valid_actions.append("BOMB")

    return valid_actions or ["WAIT"]


def _is_free_tile(field, position, bomb_positions, enemy_positions):
    x, y = position
    if x < 0 or y < 0 or x >= field.shape[0] or y >= field.shape[1]:
        return False
    return (
        field[x, y] == 0
        and position not in bomb_positions
        and position not in enemy_positions
    )


def _can_place_useful_safe_bomb(game_state):
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    _, _, bomb_available, position = game_state["self"]

    if not bomb_available:
        return False

    if is_position_danger(field, explosion_map, position, bombs):
        return False

    hypothetical_bombs = list(bombs)
    hypothetical_bombs.append((position, 4))
    if not can_escape_from(position, field, hypothetical_bombs, others, explosion_map):
        return False

    return _bomb_has_target(field, position, others)


def _bomb_has_target(field, position, others):
    x, y = position
    enemy_positions = {agent[3] for agent in others}

    for dx, dy in MOVE_DELTAS.values():
        for distance in range(1, 4):
            tx = x + dx * distance
            ty = y + dy * distance
            tile = field[tx, ty]

            if tile == -1:
                break
            if tile == 1:
                return True
            if (tx, ty) in enemy_positions:
                return True

    return False



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

    bomb_positions = {bomb_pos for bomb_pos, timer in game_state["bombs"]}
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

    if coins:
        # finding nearest coin using manhattan distance
        nearest_coin = min(
            coins,
            key=lambda coin: abs(coin[0] - x) + abs(coin[1] - y)
        )

        coin_x, coin_y = nearest_coin

        dx = coin_x - x
        dy = coin_y - y

        # the one-hot direction of the nearest coin
        features.append(int(dy < 0))  # up
        features.append(int(dx > 0))  # right
        features.append(int(dy > 0))  # down
        features.append(int(dx < 0))  # left

    else:
        # no coins exist
        # no direction
        features.extend([0, 0, 0, 0])


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

    # danger indicators
    # danger is defined as being in the blast radius of a bomb
    bombs = game_state["bombs"]
    danger_up = is_position_danger(
        field, explosion_map, (x, y - 1), bombs)

    danger_right = is_position_danger(
        field, explosion_map, (x+1, y), bombs
    )

    danger_down = is_position_danger(
        field, explosion_map, (x, y + 1), bombs
    )

    danger_left = is_position_danger(
        field, explosion_map, (x-1, y), bombs
    )

    features.append(int(danger_up))
    features.append(int(danger_right))
    features.append(int(danger_down))
    features.append(int(danger_left))

    # current danger

    current_danger = is_position_danger(
        field, explosion_map, (x, y), bombs
    )

    features.append(int(current_danger))

    # escape
    escape_up = can_escape_from(
        (x, y - 1),
        field,
        bombs,
        other_agents,
        explosion_map
    )

    escape_right = can_escape_from(
        (x + 1, y),
        field,
        bombs,
        other_agents,
        explosion_map
    )

    escape_down = can_escape_from(
        (x, y + 1),
        field,
        bombs,
        other_agents,
        explosion_map
    )

    escape_left = can_escape_from(
        (x - 1, y),
        field,
        bombs,
        other_agents,
        explosion_map
    )


    features.append(int(escape_up))
    features.append(int(escape_right))
    features.append(int(escape_down))
    features.append(int(escape_left))



    features = np.asarray(features, dtype=np.float32)

    return torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)


def is_position_danger(field, explosion_map, position, bombs):
    x, y = position

    # already in a blast
    if explosion_map[x, y] > 0:
        return True

    for bomb in bombs:
        (bomb_x, bomb_y), timer = bomb

        # Bomb's own position is dangerous
        if (x, y) == (bomb_x, bomb_y):
            return True

        # same row
        if x == bomb_x:
            distance = abs(bomb_y - y)
            if distance <= 3:
                blocked = False

                step = 1 if y > bomb_y else -1

                for d in range(1, distance + 1):
                    cell_y = bomb_y + d * step

                    if field[x, cell_y] == -1:
                        blocked = True
                        break

                    if field[x, cell_y] == 1:
                        blocked = True
                        break

                if not blocked:
                    return True

        # same column
        if y == bomb_y:
            distance = abs(bomb_x - x)
            if distance <= 3:
                blocked = False

                step = 1 if x > bomb_x else -1

                for d in range(1, distance + 1):
                    cell_x = bomb_x + d * step

                    if field[cell_x, y] == -1:
                        blocked = True
                        break

                    if field[cell_x, y] == 1:
                        blocked = True
                        break

                if not blocked:
                    return True

    return False


def can_escape_from(start, field, bombs, others, explosion_map, max_depth=6):
    """
    this is a BFS search to find if there is a path from the current position to a safe position

    start : tuple[int, int]
        position to start the search from

    others -> game_state["others"]

    """
    enemy_positions = {agent[3] for agent in others}

    bomb_positions = {bomb_pos for bomb_pos, timer in bombs}

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

    # is the position inside future bomb's blast radius

    def in_future_blast(position):
        px, py = position
        for (bomb_x, bomb_y), timer in bombs:

            #bomb pos self
            if position == (bomb_x, bomb_y):
                return True

            # same x coordinate
            if px == bomb_x:
                distance = abs(py - bomb_y)
                if 1 <= distance <= 3:

                    direction = 1 if py > bomb_y else -1
                    blocked = False

                    for d in range(1, distance + 1):

                        check_y = bomb_y + d * direction

                        # stone wall blocks explosion
                        if field[px, check_y] == -1:
                            blocked = True
                            break

                        # crate blocks explosion
                        if field[px, check_y] == 1:
                            blocked = True
                            break

                    if not blocked:
                        return True

            # same y coordinate
            if py == bomb_y:

                distance = abs(px - bomb_x)

                if 1 <= distance <= 3:

                    direction = 1 if px > bomb_x else -1
                    blocked = False

                    for d in range(1, distance + 1):

                        check_x = bomb_x + direction * d

                        if field[check_x, bomb_y] == -1:
                            blocked = True
                            break

                        if field[check_x, bomb_y] == 1:
                            blocked = True
                            break

                    if not blocked:
                        return True

        return False

    # is this tile currently safe

    def is_safe(position):
        px, py = position

        if not is_walkable(position):
            return False

        # active explosion
        if explosion_map[px, py] > 0:
            return False

        # future bomb blast
        if in_future_blast(position):
            return False

        return True

    # BFS search for a safe position

    queue = deque()
    queue.append((start, 0))

    visited = {start}

    while queue:

        current_position, depth = queue.popleft()
        # depth > 0 should be to prevent the start position that itself is safe
        if depth > 0 and is_safe(current_position):
            return True

        if depth >= max_depth:
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

            # not walking into an explosion that is already active
            nx, ny = next_position

            if explosion_map[nx, ny] > 0:
                continue

            visited.add(next_position)

            queue.append((next_position, depth + 1))

    return False
