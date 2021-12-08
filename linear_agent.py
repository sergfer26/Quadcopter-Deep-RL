import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from env import QuadcopterEnv, AgentEnv
from DDPG.utils import OUNoise
from simulation import sim, nSim, nSim3D
from Linear.step import control_feedback  # F, C
from Linear.constants import CONSTANTS, F, C


G = CONSTANTS['G']
M = CONSTANTS['M']
K = CONSTANTS['K']
B = CONSTANTS['B']
L = CONSTANTS['L']
Ixx = CONSTANTS['Ixx']
Iyy = CONSTANTS['Iyy']
Izz = CONSTANTS['Izz']


F1, F2, F3, F4 = F
c1, c2, c3, c4 = C


class LinearAgent:

    def __init__(self, env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

    def get_action(self, state):
        u, v, w, x, y, z, p, q, r, psi, theta, phi = state
        W1 = control_feedback(z, w, F1) * (c1 ** 2)  # control z
        W2 = control_feedback(psi, r, F2) * c2  # control yaw
        W3 = control_feedback(phi, p, F3) * c3  # control roll
        W4 = control_feedback(theta, q, F4) * c4  # control pitch
        W = W1 + W2 + W3 + W4
        return W.reshape(4)


class LinearEnv(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.noise = OUNoise(env.action_space)
        self.noise_on = True

    def action(self, action):
        if self.noise_on:
            action = self.noise.get_action(action, self.i)
        return action


'''
if __name__ == "__main__":
    env = QuadcopterEnv(reward='r4')
    env = LinearEnv(env)
    env.noise_on = False
    agent = LinearAgent(env)
    states, actions, scores = sim(True, agent, env)

    steps = states.shape[0]
    fig1, axs1 = plt.subplots(agent.num_states // 2, 2)
    names = (r'$u$', r'$v$', r'$w$', r'$x$', r'$y$', r'$z$', r'$p$',
             r'$q$', r'$r$', r'$\psi$', r'$\theta$', r'$\phi$')
    data = pd.DataFrame(states, columns=names)
    data['$t$'] = env.time[0:steps]
    data.plot(x='$t$', subplots=True, ax=axs1, legend=True)
    plt.show()

    steps = actions.shape[0]
    fig2, axs2 = plt.subplots(agent.num_actions // 2, 2)
    names = (r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$')
    data = pd.DataFrame(actions, columns=names)
    data['$t$'] = env.time[0:steps]
    data.plot(x='$t$', subplots=True, ax=axs2, legend=True)
    plt.show()

    steps = scores.shape[0]
    fig3, axs3 = plt.subplots(4 // 2, 2)
    names = (r'$r_t$', r'$Cr_t$', 'stable', 'contained')
    data = pd.DataFrame(scores, columns=names)
    data['$t$'] = env.time[0:steps]
    data.plot(x='$t$', subplots=True, ax=axs3, legend=True)
    plt.show()
'''
