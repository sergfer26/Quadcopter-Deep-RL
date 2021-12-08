import numpy as np
import pandas as pd
from env import QuadcopterEnv
from matplotlib import pyplot as plt
from simulation import sim
from linear_agent import LinearEnv
from Linear.constants import CONSTANTS, omega_0

#    z, phi, theta, psi
KD = np.array([2.5, 1.75, 1.75, 1.75])
KP = np.array([1.5, 6, 6, 6])

G = CONSTANTS['G']
M = CONSTANTS['M']
K = CONSTANTS['K']
B = CONSTANTS['B']
L = CONSTANTS['L']
Ixx = CONSTANTS['Ixx']
Iyy = CONSTANTS['Iyy']
Izz = CONSTANTS['Izz']

W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0


def tau2omega(t, x, y):
    aux = 0.0
    aux += t/(4 * K)
    aux += x/(2 * K * L)
    aux += y/(4 * B)
    return aux


def pid(kp, kd, c, x, xf, x_dot, xf_dot, dt):
    error = xf - x
    de = xf_dot - x_dot
    return (kd * de + kp * error) * c


class PIDAgent:

    def __init__(self, env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.dt = env.time[1] - env.time[0]
        self.goal = np.zeros(4)
        self.goal_dot = np.zeros(4)
        self.reset()

    def get_action(self, state):
        u, v, w, x, y, z, p, q, r, psi, theta, phi = state

        c1 = M/(np.cos(phi) * np.cos(theta))
        #b1 = G * c1
        #C = np.array([c1, Ixx, Iyy, Izz])
        #X = np.array([z, phi, theta, psi])
        #X_dot = np.array([w, p, q, r])
        '''
        tau = pid(KP, KD, C, X, self.goal, X_dot,
                  self.goal_dot, self.dt)  # thrust, torques
        tau[0] += b1
        '''
        T = (G + KD[0] * w + KP[0] * z) * c1
        tau_phi = (KD[1] * p + KP[1] * phi) * Ixx
        tau_theta = (KD[2] * q + KP[2] * theta) * Iyy
        tau_psi = (KD[2] * r + KP[2] * psi) * Izz
        tau = np.array([T, tau_phi, tau_theta, tau_psi])
        return self.get_control(tau)

    def get_control(self, tau):
        t, tau_phi, tau_theta, tau_psi = tau
        w1 = tau2omega(t, - tau_theta, - tau_psi)
        w2 = tau2omega(t, - tau_phi, + tau_psi)
        w3 = tau2omega(t, + tau_theta, - tau_psi)
        w4 = tau2omega(t, + tau_phi, + tau_psi)
        w = np.array([w1, w2, w3, w4])
        return np.sqrt(w)

    def reset(self):
        pid.old_error = np.full(4, None)


if __name__ == "__main__":
    env = QuadcopterEnv(reward='r4')
    env = LinearEnv(env)
    env.noise_on = False
    agent = PIDAgent(env)
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
