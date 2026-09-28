from collections import deque, namedtuple

import numpy as np


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward", "next_action_mask"),
)


class Replay_memory(object):
    """Proportional prioritized experience replay.

    Transitions with larger absolute TD errors are sampled more often. The
    returned importance-sampling weights correct the resulting training bias.
    """

    def __init__(self, capacity, alpha=0.6, priority_epsilon=1e-5):
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.max_priority = 1.0

    def push(self, *args):
        """Save one transition with the maximum priority observed so far.

        Giving new transitions maximum priority ensures they are sampled at
        least once before their true TD-error priority is known.
        """
        self.memory.append(Transition(*args))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta=0.4):
        if batch_size > len(self.memory):
            raise ValueError("batch_size cannot exceed replay-memory size")

        priorities = np.asarray(self.priorities, dtype=np.float64)
        scaled_priorities = np.power(priorities, self.alpha)
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(
            len(self.memory),
            size=batch_size,
            replace=True,
            p=probabilities,
        )

        transitions = [self.memory[index] for index in indices]

        weights = np.power(len(self.memory) * probabilities[indices], -beta)
        weights /= weights.max()

        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update sampled transitions from their latest absolute TD errors."""
        for index, td_error in zip(indices, td_errors):
            priority = abs(float(td_error)) + self.priority_epsilon
            self.priorities[int(index)] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)
