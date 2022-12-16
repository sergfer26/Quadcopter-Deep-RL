import random
import gym
import torch
import numpy as np
from collections import deque


class Memory:

    def __init__(self, max_size, action_dim, state_dim, T):
        self.max_size = max_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.T = T
        self.buffer = deque(maxlen=max_size)

    def push(self, trajectory=None, **kwargs):
        '''
        Método para agregar trayectorias al buffer de la memoria.

        Argumentos
        ----------
        trajectory (ocional): list
            La lista con las tuplas que representan la experiencia a lo largo
            de un episodio. Cada tupla debe tener la siguiente estructura:
            (state, action, new_state, done). `len(trajectory) -> self.T`.
        states : `np.ndarray`
            Es un arreglo que representa los estados de una o `m` simulaciones.
            `states.shape -> (self.T, self.state_dim)`
            o `states.shape -> (m, self.T, self.state_dim)`
        actions : `np.ndarray`
            Es un arrreglo que representa las acciones de una o `m`
            simulaciones.
            `actions.shape -> (self.T, self.action_dim)`
            o `states.shape -> (m, self.T, self.action_dim)`
        next_states : `np.ndarray`
            Es un arreglo que representa los nuevos estados de una o `m`
            simulaciones.
            `next_states.shape -> (self.T, self.state_dim)`
            o `next_states.shape -> (m, self.T, self.state_dim)`
        dones : `np.array`
            Es un arreglo que representa los booleanos de termino de una
            simulación o `m` simulaciones.
            `dones.shape -> self.T` o `next_states.shape -> (m, self.T)`
        '''
        if isinstance(trajectory, list):
            states, actions, next_states, dones = self._preprocess_traj(
                trajectory)
        elif len(kwargs) > 0:
            states = kwargs['states']
            actions = kwargs['actions']
            next_states = kwargs['next_states']
            dones = kwargs['dones']
        if len(states.shape) == 2:
            x = [states, actions, next_states, dones]
            self.buffer.append(x)
        elif len(states.shape) == 3:
            for i in range(states.shape[0]):
                x = [states[i, :, :], actions[i, :, :],
                     next_states[i, :, :], dones[i, :]]
                self.buffer.append(x)
        else:
            print('No pudo procesar la información')

    def sample(self, sample_size, t_x=None, t_u=None):
        '''
        Muestrea trayectorias de simulaciones.

        Argumentos
        ----------
        sample_size : int
            Representa el tamaño de muestra.

        Retornos
        --------
        states : `np.ndarray`
            Arreglo que representa los estados de `sample_size` simulaciones.
            `states.shape -> (sample_size, self.T, self.state_dim)`.
        actions : `np.ndarray`
            Arreglo que representa las acciones de `sample_size` simulaciones.
            `states.shape -> (sample_size, self.T, self.action_dim)`.
        new_states : `np.ndarray`
            Arreglo que representa los nuevos estados de `sample_size`
            simulaciones.
            `states.shape -> (sample_size, self.T, self.state_dim)`.
        dones : `np.ndarray`
            Arreglo que representa los booleanos de terminado de
            `sample_size` simulaciones.
            `dones.shape -> (sample_size, self.T)`.
        '''
        samples = random.sample(self.buffer, sample_size)
        states = np.empty((sample_size, self.T, self.state_dim))
        actions = np.empty((sample_size, self.T, self.action_dim))
        new_states = np.empty((sample_size, self.T, self.state_dim))
        dones = np.empty((sample_size, self.T))
        for i in range(sample_size):
            states[i, :, :] = samples[i][0]
            actions[i, :, :] = samples[i][1]
            new_states[i, :, :] = samples[i][2]
            dones[i, :] = samples[i][3]
        if callable(t_x):
            states = np.apply_along_axis(t_x, -1, states)
            new_states = np.apply_along_axis(t_x, -1, new_states)
        if callable(t_u):
            actions = np.apply_along_axis(t_u, -1, actions)
        return states, actions, new_states, dones

    def _preprocess_traj(self, trajectory):
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


class CostNN_ENV(gym.Wrapper):

    def __init__(self, env, cost):
        super().__init__(env)
        self.cost = cost
        self.observation = None

    def set_cost(self, cost):
        self.cost = cost.eval()

    def reward(self, state, action):
        self.cost.training = False
        z = np.hstack([state, action])
        z = torch.tensor(z, dtype=torch.float32)
        return - self.cost(z).item

    def step(self, action):
        state = self.observation
        observation, _, done, info = self.env.step(action)
        self.observation = observation
        return observation, self.reward(state, action), done, info

    def reset(self):
        self.observation = super().reset()
        return self.observation
