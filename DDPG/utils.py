import numpy as np
import gym
import random
import torch
from numpy import floor
from collections import deque
from .params import PARAMS_UTILS


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=PARAMS_UTILS['mu'],
                 theta=PARAMS_UTILS['theta'],
                 max_sigma=PARAMS_UTILS['max_sigma'],
                 min_sigma=PARAMS_UTILS['min_sigma'],
                 decay_period=PARAMS_UTILS['decay_period']):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return np.clip(self.state, self.low, self.high)

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def __init__(self, env):
        super().__init__(env)
        # QuadcopterEnv.__init__(self)

    def action(self, action):
        '''
        se alimenta de la tanh
        '''
        high = self.action_space.high
        low = self.action_space.low
        if torch.is_tensor(action):
            high = torch.tensor(high)
            low = torch.tensor(low)

        act_k = (high - low) / 2.
        act_b = (high + low) / 2.
        try:
            aux = act_k * action + act_b
        except:
            breakpoint()
        return aux

    def reverse_action(self, action):
        high = self.action_space.high
        low = self.action_space.low
        if torch.is_tensor(action):
            high = torch.tensor(high)
            low = torch.tensor(low)

        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class AgentEnv(NormalizedEnv):

    def __init__(self, env, tx=None, inv_tx=None):
        super().__init__(env)
        self.noise = OUNoise(env.action_space)
        self.noise_on = True
        self._tx = tx
        self._inv_tx = inv_tx
        low = self._tx(env.observation_space.low)
        high = self._tx(env.observation_space.high)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)
        self.i = 0

    def action(self, action):
        action = super().action(action)
        if self.noise_on:
            action = self.noise.get_action(action, self.i)
            self.i += 1
        return action

    def observation(self, state):
        if callable(self._tx):
            state = self._tx(state)
        return state

    def reverse_observation(self, state):
        if callable(self._inv_tx):
            state = self._inv_tx(state)
        return state

    def step(self, action):
        new_state, reward, done, info = super().step(action)
        action = self.reverse_action(info['real_action'])
        info['action'] = action
        new_state = self.observation(new_state)
        return new_state, reward, done, info

    def reset(self):
        state = super().reset()
        return self.observation(state)


class Memory:

    def __init__(self, max_size, n_x, n_u):
        self.max_size = max_size
        self.n_x = n_x
        self.n_u = n_u
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = np.empty((batch_size, self.n_x))
        action_batch = np.empty((batch_size, self.n_u))
        reward_batch = np.empty(batch_size)
        next_state_batch = np.empty((batch_size, self.n_x))
        done_batch = np.empty(batch_size)

        batch = random.sample(self.buffer, batch_size)

        for i, experience in enumerate(batch):
            state, action, reward, next_state, done = experience
            state_batch[i] = state
            action_batch[i] = action
            reward_batch[i] = reward
            next_state_batch[i] = next_state
            done_batch[i] = done

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    def remove(self):
        n = int(floor(len(self.buffer)/2))
        for _ in range(n):
            i = np.random.randint(n)
            del self.buffer[i]
        print('Se elimino el 50% de la infomacion!!')
