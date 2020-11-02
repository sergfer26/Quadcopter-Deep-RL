import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from DDPG.env.quadcopter_env import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, XE, YE, funcion
from Linear.step import control_feedback, F1, F2, F3, F4, c1, c2, c3, c4
from DDPG.utils import NormalizedEnv, OUNoise
from DDPG.ddpg import DDPGagent
from numpy.linalg import norm
from DDPG.load_save import load_nets, save_nets, remove_nets, save_buffer, remove_buffer
from tools.tools import imagen2d, imagen_action, sub_plot_state
from numpy import pi, cos, sin
from numpy import remainder as rem


BATCH_SIZE = 32
TAU = 2 * pi
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0


def agent_action(agent, env, noise, state):
    lambdas, action = agent.get_action(state)
    real_action = env._action(action)
    real_action = noise.get_action(real_action, env.time[env.i])
    control = W0 + real_action
    new_state, reward, done = env.step(control)
    return real_action, action, lambdas, new_state, reward, done


def linear_action(env, state):
    _, _, w, _, _, z, p, q, r, psi, theta, phi = state
    W1 = control_feedback(z - env.goal[5], w, F1) * c1  # control z
    W2 = control_feedback(psi - env.goal[-3], r, F2) * c2  # control yaw
    W3 = control_feedback(phi - env.goal[-1], p, F3) * c3  # control roll
    W4 = control_feedback(theta - env.goal[-2], q, F4) * c4  # control pitch
    action = W1 + W2 + W3 + W4
    action = action.reshape(4)
    control = W0 + action
    _, reward, done = env.step(control)
    return action, reward, done


def training_loop(agent, env, noise, pbar=None):
    state = funcion(env.reset())
    noise.reset()
    episode_reward = 0
    s = 1
    while True:
        _, _, lambdas, new_state, reward, done = agent_action(agent, env, noise, state)
        episode_reward += reward
        s1, s2 = env.get_score(env.state)
        agent.memory.push(state, lambdas, reward, new_state, done)
        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)
        _, _, w, _, _, z, p, q, _, _, theta, phi = env.state
        if pbar:
            pbar.set_postfix(R='{:.2f}'.format(episode_reward),
                w='{:.2f}'.format(w), p='{:.2f}'.format(p), q='{:2f}'.format(q),
                    theta='{:.2f}'.format(rem(theta, TAU)), phi='{:.2f}'.format(rem(phi, TAU)), 
                        z='{:.2f}'.format(z), s='{:.4f}'.format(noise.max_sigma))
            pbar.update(1)
        if done:
            break
        state = new_state
        s += 1
    return s1/s, s2/s, episode_reward/s


def Sim(flag, agent, env, noise, show=True, path=None):
    t = env.time
    state = funcion(env.reset())
    noise.reset()
    u, v, w, x, y, z, p, q, r, psi, theta, phi  = env.state
    episode_reward = 0
    Z, W, Psi, R, Phi, P, Theta, Q = [z], [w], [psi], [r], [phi], [p], [theta], [q]
    X, U, Y, V = [x], [u], [y], [v]
    acciones = []
    env.flag  = flag
    while True:
        real_action, _, _, new_state, reward, done = agent_action(agent, env, noise, state)
        u, v, w, x, y, z, p, q, r, psi, theta, phi = env.state
        Z.append(z); W.append(w)
        Psi.append(psi); R.append(r)
        Phi.append(phi); P.append(p)
        Theta.append(theta); Q.append(q)
        X.append(x); Y.append(y)
        U.append(u); V.append(v)
        acciones.append(real_action)
        state = new_state
        episode_reward += reward
        if done:
            break
    T = t[0:len(Z)]
    S = (U, V, W, X, Y, Z, P, Q, R, Psi, Theta, Phi) 
    imagen2d(S, T, show=show, path=path)
    imagen_action(acciones, T, show=show, path=path)


def plot_agent_vs_linear(X, Y, t, title=None, show=True, path=None):
    '''
    X -> linear states np array, shape = (n, 12)
    Y -> agent states np array, shape = (m, 12)
    '''
    fig, (aX, adX, aPhi, adPhi) = plt.subplots(4, 3)
    u, v, w, x, y, z = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    p, q, r, psi, theta, phi = X[:, 6], X[:, 7], X[:, 8], X[:, 9], X[:, 10], X[:, 11]
    T = t[0: len(z)]

    u_, v_, w_, x_, y_, z_ = X[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4], Y[:, 5]
    p_, q_, r_, psi_, theta_, phi_ = Y[:, 6], Y[:, 7], Y[:, 8], Y[:, 9], Y[:, 10], Y[:, 11]
    T_ = t[0: len(z_)]

    labels = ('linear', None, None)
    sub_plot_state(T, x, y, z, aX, axis_labels=['x', 'y', 'z'], c=['b', 'b', 'b'], labels=labels)
    sub_plot_state(T_, x_, y_, z_, aX, axis_labels=['x', 'y', 'z'], c=['r', 'r', 'r'])

    sub_plot_state(T, u, v, w, adX, axis_labels=['dx', 'dy', 'dz'], c=['b', 'b', 'b'])
    labels = ('agent', None, None)
    sub_plot_state(T_, u_, v_, w_, adX, axis_labels=['dx', 'dy', 'dz'], c=['r', 'r', 'r'])

    sub_plot_state(T, psi, theta, phi, aPhi, axis_labels=['$\psi$', '$\\theta$', '$\phi$'], c=['b', 'b', 'b'])
    sub_plot_state(T_, psi_, theta_, phi_, aPhi, axis_labels=['$\psi$', '$\\theta$', '$\phi$'], c=['r', 'r', 'r'])

    sub_plot_state(T, r, q, p, adPhi, axis_labels=['$d\psi$', '$d\\theta$', '$d\phi$'], c=['b', 'b', 'b'])
    sub_plot_state(T_, r_, q_, p_, adPhi, axis_labels=['$d\psi$', '$d\\theta$', '$d\phi$'], c=['r', 'r', 'r'])

    if title:
        fig.suptitle(title)

    if show:
        plt.show()
    else:
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(path, dpi=300)
        plt.close()


def actions_agent_vs_linear(X, Y, t, title=None, show=True, path=None):
    X = np.array(X); Y = np.array(Y)
    T = t[0: len(X[:, 0])]
    T_ = t[0: len(Y[:, 0])]
    fig, ax = plt.subplots(4, 1)
    #cero = np.zeros(len(action[0])) 
    ax[0].plot(T, X[:, 0], c='b'); ax[0].plot(T_, Y[:, 0], c='r')
    ax[0].set_ylabel('$a_1$')

    ax[1].plot(T, X[:, 1], c='b'); ax[1].plot(T_, Y[:, 1], c='r')
    ax[1].set_ylabel('$a_2$')

    ax[2].plot(T, X[:, 2], c='b'); ax[2].plot(T_, Y[:, 2], c='r')
    ax[2].set_ylabel('$a_3$')

    ax[3].plot(T, X[:, 3], c='b'); ax[3].plot(T_, Y[:, 3], c='r')
    ax[3].set_ylabel('$a_4$')

    if title:
        fig.suptitle(title)

    if show:
        plt.show()
    else:
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(path, dpi=300)
        plt.close()



def agent_vs_linear(flag, agent, env, noise, show=True, paths=[None, None]):
    env.flag = flag
    t = env.time
    state = funcion(env.reset())
    noise.reset()
    env_state = env.state
    agent_states = env_state; agent_states.reshape(1, 12)
    linear_states = env_state; linear_states.reshape(1, 12)
    agent_actions = []
    linear_actions = []
    episode_reward_linear = 0.0 
    episode_reward_agent = 0.0

    while True:
        real_action, _, _, new_state, reward, done = agent_action(agent, env, noise, state)
        agent_states = np.vstack([agent_states, env.state])
        state = new_state
        episode_reward_agent += reward
        agent_actions.append(real_action)
        if done:
            break
    
    state = env_state
    env.state = state
    env.i = 0
    while True:
        action, reward, done = linear_action(env, state)
        linear_states = np.vstack([linear_states, env.state])
        state = env.state
        episode_reward_linear += reward
        linear_actions.append(action)
        if done:
            break
    
    title1 = 'Linear $G_T =$ {}, Agent $G_T =$ {}'.format(episode_reward_linear, episode_reward_agent)
    title2 = '$\sigma \max =$ {}, $\sigma \min =$ {}'.format(noise.max_sigma, noise.min_sigma)
    plot_agent_vs_linear(linear_states, agent_states, t, title=title1, show=show, path=paths[0])
    actions_agent_vs_linear(linear_actions, agent_actions, t, title=title2, show=show, path=paths[1])
    return episode_reward_agent


def nSim(flag, agent, env, noise, n, bar=None, show=True, path=None):
    # fig, (aX, aY, aZ, aPsi, aTheta, aPhi) = plt.subplots(6, 2)
    fig, (aX, adX, aPhi, adPhi) = plt.subplots(4, 3)
    alpha = 0.2
    mean_reward = 0.0
    for _ in range(n):
        if bar:
            bar.next()
        t = env.time
        state = funcion(env.reset())
        noise.reset()
        u, v, w, x, y, z, p, q, r, psi, theta, phi  = env.state
        episode_reward = 0
        Z, W, Psi, R, Phi, P, Theta, Q, T = [z], [w], [psi], [r], [phi], [p], [theta], [q], [t]
        X, U, Y, V = [x], [u], [y], [v]
        env.flag  = flag
        total = 0
        while True:
            _, _, _, new_state, reward, done = agent_action(agent, env, noise, state)
            u, v, w, x, y, z, p, q, r, psi, theta, phi = env.state
            Z.append(z); W.append(w)
            Psi.append(psi); R.append(r)
            Phi.append(phi); P.append(p)
            Theta.append(theta); Q.append(q)
            X.append(x); U.append(u)
            Y.append(y); V.append(v)
            state = new_state
            episode_reward += reward
            if done:
                _, score = env.get_score(state)
                total += score
                mean_reward += episode_reward
                break
        T = t[0:len(Z)]
        
        sub_plot_state(T, X, Y, Z, aX, axis_labels=['x', 'y', 'z'], c=['y', 'c', 'b'], alpha=alpha)
        sub_plot_state(T, U, V, W, adX, axis_labels=['dx', 'dy', 'dz'], c=['y', 'c', 'b'], alpha=alpha)
        sub_plot_state(T, Psi, Theta, Phi, aPhi, axis_labels=['$\psi$', '$\\theta$', '$\phi$'], c=['r', 'k', 'g'], alpha=alpha)
        sub_plot_state(T, R, Q, P, adPhi, axis_labels=['$d\psi$', '$d\\theta$', '$d\phi$'], c=['r', 'k', 'g'], alpha=alpha)
    
    mean_reward /= n
    title = "Vuelos terminados $f =$ {}, $G_t (media) = ${}".format(total/n, mean_reward)
    fig.suptitle(title, fontsize=12)
    if show:
        plt.show()
    else:
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(path, dpi=300)
        plt.close()

    return mean_reward

