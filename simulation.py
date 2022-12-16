from cmath import isnan
import gym
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from progressbar import progressbar


def rollout(agent, env, flag=False, state_init=None):
    # t = env.time
    env.flag = flag
    state = env.reset()
    if hasattr(agent, 'reset') and callable(getattr(agent, 'reset')):
        agent.reset()
    if isinstance(state_init, np.ndarray):
        env.state = state_init
        state = state_init
    states = np.zeros((env.steps, env.observation_space.shape[0]))
    actions = np.zeros((env.steps - 1, env.action_space.shape[0]))
    scores = np.zeros((env.steps - 1, 2))  # r_t, Cr_t
    states[0, :] = state
    episode_reward = 0
    i = 0
    while True:
        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        states[i + 1, :] = state
        if isinstance(info, dict) and ('real_action' in info.keys()):
            action = info['real_action']  # env.action(action)
        actions[i, :] = action
        scores[i, :] = np.array([reward, episode_reward])
        state = new_state
        if done:
            break
        i += 1
    return states, actions, scores


def n_rollouts(agent, env, n, flag=False, states_init=None, t_x=None, t_u=None):
    n_states = np.zeros((n, env.steps, env.observation_space.shape[0]))
    n_actions = np.zeros((n, env.steps - 1, env.action_space.shape[0]))
    n_scores = np.zeros((n, env.steps - 1, 2))
    state_init = None
    for k in range(n):  # for k in progressbar(range(n)):
        if isinstance(states_init, np.ndarray):
            if len(states_init.shape) == 2:
                state_init = states_init[k, :]
            else:
                state_init = states_init
        states, actions, scores = rollout(
            agent, env, flag=flag, state_init=state_init)
        n_states[k, :, :] = states
        n_actions[k, :, :] = actions
        n_scores[k, :, :] = scores
    if callable(t_x):
        n_states = np.apply_along_axis(t_x, -1, n_states)
    if callable(t_u):
        n_actions = np.apply_along_axis(t_u, -1, n_actions)
    return n_states, n_actions, n_scores


def n_rollouts3d(n, agent, env, path, show=False):
    flag = True
    n_states, _, _ = n_rollouts(agent, env, n)
    plot_nSim3D(n_states, path, show)


def plot_nSim3D(n_states):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    n = n_states.shape[-1]
    for k in range(n):
        states = n_states[:, :, k]
        X = states[:, 3]
        Y = states[:, 4]
        Z = states[:, 5]
        ax.plot([X[0]], [Y[0]], [Z[0]], '.c',
                alpha=0.75, label='$X_0$', markersize=5)
        ax.plot(X, Y, Z, '-b', alpha=0.1, label='$X_t$', markersize=1)
        ax.plot([X[-1]], [Y[-1]], [Z[-1]], '.r',
                alpha=0.75, label='$X_T$', markersize=5)
        if k == 0:
            ax.legend()

    fig.suptitle(r'# of flights = ' + '{} '.format(n), fontsize=12)
    # ax.plot(0, 0, 0, '.c', alpha=1, markersize=15)
    fig.set_size_inches(33., 21.)
    return fig, ax


def plot_rollouts(array: np.ndarray, time: np.ndarray, columns: list, ax=None, subplots=True, dpi=150, colors=None):
    if len(array.shape) == 2:
        array = array.reshape(1, array.shape[0], array.shape[1])
    if not isinstance(colors, list):
        colors = ['red', 'blue', 'green']
    samples, steps, n_var = array.shape
    fig = None
    if not isinstance(ax, np.ndarray) and not isinstance(ax, plt.Axes):
        if subplots:
            fig, ax = plt.subplots(n_var // 2, 2, dpi=dpi)
        else:
            fig, ax = plt.subplots(dpi=dpi)
    for k in range(samples):
        data = pd.DataFrame(array[k, :, :], columns=columns)
        data['$t (s)$'] = time[0: steps]
        if k == 0:
            legend = True
        else:
            legend = False

        data.plot(x='$t (s)$', subplots=subplots,
                  ax=ax, legend=legend, alpha=0.4)

    if not pd.isna(fig):
        fig.set_size_inches(18.5, 10.5)
    return fig, ax
