import random
import numpy as np
from collections import deque


class Memory:

    def __init__(self, max_size, action_dim, state_dim, T):
        self.max_size = max_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.T = T
        self.buffer = deque(maxlen=max_size)

    def push(self, trajectory):
        states, actions, next_states, dones = self.preprocess_traj(trajectory)
        x = [states, actions, next_states, dones]
        self.buffer.append(x)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def preprocess_traj(self, trajectory):
        states = np.zeros((self.T, self.state_dim))
        actions = np.zeros((self.T, self.action_dim))
        next_states = np.zeros((self.T, self.state_dim))
        dones = np.zeros((self.T, 1))
        for i, experience in enumerate(trajectory):
            state, action, next_state, done = experience
            states[i, :] = state
            actions[i, :] = action
            next_states[i, :] = next_state
            dones[i, :] = done
        return states, actions, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
