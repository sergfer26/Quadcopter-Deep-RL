import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sim(flag, agent, env):
    # t = env.time
    env.flag = flag
    state = env.reset()
    states = np.zeros((env.steps, len(env.state)))
    actions = np.zeros((env.steps - 1, env.action_space.shape[0]))
    scores = np.zeros((env.steps - 1, 4))  # r_t, Cr_t, stable, contained
    states[0, :] = env.state
    episode_reward = 0
    i = 0
    while True:
        action = agent.get_action(state)
        action, reward, new_state, done = env.step(action)
        episode_reward += reward
        states[i + 1, :] = env.state
        actions[i, :] = env.action(action)
        scores[i, :] = np.array([reward, episode_reward, env.is_stable(
            new_state), env.is_contained(new_state)])
        state = new_state
        if done:
            break
        i += 1
    return states, actions, scores


def nSim(flag, agent, env, n):
    n_states = np.zeros((env.steps, len(env.state), n))
    n_actions = np.zeros((env.steps - 1, env.action_space.shape[0], n))
    n_scores = np.zeros((env.steps - 1, 4, n))
    for k in range(n):
        states, actions, scores = sim(flag, agent, env)
        n_states[:, :, k] = states
        n_actions[:, :, k] = actions
        n_scores[:, :, k] = scores
    return n_states, n_actions, n_scores


def nSim3D(n, agent, env, path, show=False):
    flag = True
    n_states, _, _ = nSim(flag, agent, env, n)
    plot_nSim3D(n_states, path, show)


def plot_nSim3D(n_states, show=False, file_name=None):
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
    if show:
        plt.show()
    else:
        fig.set_size_inches(33., 21.)
        plt.savefig(file_name, dpi=300)


def plot_nSim2D(array3D, columns, time, show=True, file_name=None):
    steps, var, samples = array3D.shape
    fig, axes = plt.subplots(var // 2, 2)
    for k in range(samples):
        data = pd.DataFrame(array3D[:, :, k], columns=columns)
        data['$t$'] = time[0: steps]
        if k == 0:
            legend = True
        else:
            legend = False

        data.plot(x='$t$', subplots=True, ax=axes, legend=legend)
    fig.set_size_inches(18.5, 10.5)
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300)
