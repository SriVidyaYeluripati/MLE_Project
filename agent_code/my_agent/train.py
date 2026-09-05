from collections import namedtuple, deque

import torch
import torch.nn as nn
from typing import List

from .model import DQN
import events as e

from .callbacks import (
    ACTIONS,
    N_OBSERVATIONS,
    state_to_features,
    is_position_danger,
    can_escape_from,
)
from .memory import Replay_memory

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
UNSAFE_BOMB = "UNSAFE_BOMB"

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
NEW_POSITION = "NEW_POSITION"
REPEATED_POSITION = "REPEATED_POSITION"

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 3e-4

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    self.visited_positions = set()
    self.recent_positions = deque(maxlen=20)

    self.target_net = DQN(n_observations=N_OBSERVATIONS, n_actions=len(ACTIONS)).to(device)

    self.memory = Replay_memory(5000)

    self.optimizer = torch.optim.AdamW(
        self.policy_net.parameters(),
        lr=LR,
        amsgrad=True
    )

    self.target_net.load_state_dict(
        self.policy_net.state_dict()
    )

    self.target_net.eval()


def optimize_model(self):

    if len(self.memory) < BATCH_SIZE:
        return


    transitions = self.memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device,
        dtype=torch.bool
    )

    non_final_next_states_list = [
        s for s in batch.next_state
        if s is not None
    ]

    # state action reward batches
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    state_action_values = (
        self.policy_net(state_batch)
        .gather(1, action_batch)
    )

    # compute max_a Q_target
    next_state_values = torch.zeros(
        BATCH_SIZE,
        device=device
    )

    with torch.no_grad():

        # Only run the target network if we actually have non-terminal next states.
        if non_final_next_states_list:

            non_final_next_states = torch.cat(
                non_final_next_states_list
            )

            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states)
                .max(dim=1)
                .values
            )


    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # in-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

    # Soft update of the target network
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()

    for key in policy_net_state_dict:
        target_net_state_dict[key] = (
            policy_net_state_dict[key] * TAU
            + target_net_state_dict[key] * (1.0 - TAU)
        )

    self.target_net.load_state_dict(target_net_state_dict)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py

    state = state_to_features(old_game_state)
    next_state = state_to_features(new_game_state)

    action = torch.tensor(
        [[ACTIONS.index(self_action)]],
        dtype=torch.long,
        device=device
    )

    old_field = old_game_state["field"]
    old_explosion_map = old_game_state["explosion_map"]
    old_bombs = old_game_state["bombs"]

    new_field = new_game_state["field"]
    new_explosion_map = new_game_state["explosion_map"]
    new_bombs = new_game_state["bombs"]

    old_x, old_y = old_game_state["self"][3]
    new_x, new_y = new_game_state["self"][3]

    old_position = (old_x, old_y)
    new_position = (new_x, new_y)

    # reward exploration
    if new_position not in self.visited_positions:
        events.append(NEW_POSITION)
        self.visited_positions.add(new_position)

    # penalize repeated visited positions
    if new_position in self.recent_positions:
        events.append(REPEATED_POSITION)

    self.recent_positions.append(new_position)

    # reward movement toward the nearest visible coin
    coins = old_game_state["coins"]
    if coins:
        nearest_coin = min(
            coins, 
            key=lambda coin: abs(coin[0] - old_x) + abs(coin[1] - old_y)
        )

        old_distance = (
            abs(nearest_coin[0] - old_x) + abs(nearest_coin[1] - old_y)
        )

        new_distance = (
            abs(nearest_coin[0] - new_x) + abs(nearest_coin[1] - new_y)
        )

        if new_distance < old_distance:
            events.append(MOVED_TOWARD_COIN)
        elif new_distance > old_distance:
            events.append(MOVED_AWAY_FROM_COIN)


    old_danger = is_position_danger(
        old_field,
        old_explosion_map,
        (old_x, old_y),
        old_bombs
    )

    new_danger = is_position_danger(
        new_field,
        new_explosion_map,
        (new_x, new_y),
        new_bombs
    )

    if old_danger and not new_danger:
        events.append(ESCAPED_DANGER)

    if (
    not old_danger
    and new_danger
    and self_action != "BOMB"):
        events.append(ENTERED_DANGER)

    # Penalize dropping a bomb when there is no escape route
    if self_action == "BOMB":

        bomb_available = old_game_state["self"][2]

        # Only evaluate if the bomb action was actually possible
        if bomb_available:

            x, y = old_game_state["self"][3]

            hypothetical_bombs = list(old_game_state["bombs"])

            # Simulate the bomb that we are considering placing
            hypothetical_bombs.append(
                ((x, y), 3)
            )

            safe_to_escape = can_escape_from(
                (x, y),
                old_game_state["field"],
                hypothetical_bombs,
                old_game_state["others"],
                old_game_state["explosion_map"]
            )

            if not safe_to_escape:
                events.append(UNSAFE_BOMB)

    reward = reward_from_events(self, events)

    self.memory.push(
        state,
        action,
        next_state,
        reward
    )

    optimize_model(self)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.memory.push(
        state_to_features(last_game_state),
        torch.tensor(
            [[ACTIONS.index(last_action)]],
            dtype=torch.long,
            device=device
        ),
        None,
        reward_from_events(self, events)
    )

    optimize_model(self)

    torch.save(
        self.policy_net.state_dict(),
        "my-saved-model.pt"
    )

    self.visited_positions.clear()
    self.recent_positions.clear()



def reward_from_events(self, events: List[str]) -> torch.Tensor:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.COIN_FOUND: 5,
        e.CRATE_DESTROYED: 8,
        e.KILLED_OPPONENT: 30,

        e.WAITED: -1,
        e.INVALID_ACTION: -5,
        e.KILLED_SELF: -30,
        e.GOT_KILLED: -25,

        e.SURVIVED_ROUND: 1,

        ESCAPED_DANGER: 5,
        ENTERED_DANGER: -5,
        UNSAFE_BOMB: -15,

        MOVED_TOWARD_COIN: 1,
        MOVED_AWAY_FROM_COIN: -0.5,
        NEW_POSITION: 0.5,
        REPEATED_POSITION: -0.5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return torch.tensor([reward_sum], dtype=torch.float32, device=device)
