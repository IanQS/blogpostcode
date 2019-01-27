"""
Utilities for the agent
    - replay buffer
    - OU noise
    - etc

Author: Ian Q.

Notes:

"""

from collections import deque
import numpy as np
import numpy.random as nr
from operator import itemgetter

class ReplayBuffer():
    def __init__(self, size = 10000):
        self.limit = size
        self.buffer = deque([], maxlen=self.limit)

    def sample(self, batch_size):
        random_indices = np.random.choice(np.arange(0, len(self.buffer)), size=batch_size, replace=False)
        return itemgetter(*(random_indices.tolist()))(self.buffer)

    def insert(self, transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer = deque([], maxlen=self.limit)


class OUNoise():
    # --------------------------------------
    # Ornstein-Uhlenbeck Noise
    # Author: Flood Sung
    # Date: 2016.5.4
    # Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
    # --------------------------------------

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state