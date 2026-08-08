import math
import os
import random

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
N_OBSERVATIONS = 9

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

    if self.train and random.random() < eps_threshold:
        action = random.randrange(len(ACTIONS))
    else:
        with torch.no_grad():
            q_values = self.policy_net(features)
            # print("Features:", features.cpu().numpy())
            # print("Q-values:", q_values.cpu().numpy())
            # print("Chosen:", ACTIONS[q_values.argmax(dim=1).item()])
        action = q_values.argmax(dim=1).item()

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

    _, _, bomb_available, (x, y) = game_state["self"]

    features = []

    # adjacent tiles 
    features.append(int(field[x, y - 1] == 0)) # up
    features.append(int(field[x + 1, y] == 0)) # right 
    features.append(int(field[x, y + 1] == 0)) # down
    features.append(int(field[x - 1, y] == 0)) # left

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


    features = np.asarray(features, dtype=np.float32)

    return torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)
