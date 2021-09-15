import numpy as np
import pandas as pd
from env import QuadcopterEnv
from Linear.equations import I, G
from matplotlib import pyplot as plt
from simulation import sim

#    z, phi, theta, psi
KD = np.array([2.5, 1.75, 1.75, 1.75])
KP = np.array([1.5, 6, 6, 6])
M = 0.468
K = 2.980 * 10 **−6
B = 1.140 * 10 **−7
L = 0.225
Ixx, Iyy, Izz = I


def tau2omega(t, x, y):
    aux = 0.0
    aux += t/(4 * K)
    aux += x/(2 * K * L)
    aux += y/(4 * B)
    return aux


def pid(kp, kd, c, x, xf, dt):
    error = xf - x
    if pid.old_error.all() is None:
        pid.old_error = error

    de = (error - pid.old_error) / dt

    return (kd * de + kp * error) * c


class PIDAgent:

    def __init__(self, env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.dt = env.time[1] - env.time[0]
        self.goal = np.zeros(4)
        self.reset()

    def get_action(self, state):
        u, v, w, x, y, z, p, q, r, psi, theta, phi = state
        c1 = M/(np.cos(phi) * np.cos(theta))
        b1 = G * c1
        C = np.array([c1, Ixx, Iyy, Izz])
        X = np.array([z, phi, theta, psi])
        tau = pid(KP, KD, C, X, self.goal, self.dt)  # thrust, torques
        tau[0] += b1
        return self.get_control(tau)

    def get_control(self, tau):
        t, tau_phi, tau_theta, tau_psi = tau
        w1 = np.sqrt(tau2omega(t, - tau_theta, - tau_psi))
        w2 = np.sqrt(tau2omega(t, - tau_phi, + tau_psi))
        w3 = np.sqrt(tau2omega(t, + tau_theta, - tau_psi))
        w4 = np.sqrt(tau2omega(t, + tau_phi, + tau_psi))
        return np.array([w1, w2, w3, w4])

    def reset(self):
        pid.old_error = np.full(4, None)


if __name__ == "__main__":
    env = QuadcopterEnv(reward='r4')
    # env = LinearEnv(env)
    # env.noise_on = False
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
