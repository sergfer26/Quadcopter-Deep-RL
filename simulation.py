import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from progress.bar import Bar


def sim(flag, agent, env):
    #t = env.time
    env.flag  = flag
    state = env.reset()
    states = np.zeros((int(env.steps), env.observation_space.shape[0] - 6))
    actions = np.zeros((int(env.steps), env.action_space.shape[0]))
    scores = np.zeros((int(env.steps), 4)) # r_t, Cr_t, stable, contained
    #states[0, :]= env.reverse_observation(state)
    episode_reward = 0
    i = 0
    while True:
        action = agent.get_action(state)
        action, reward, new_state, done = env.step(action)
        episode_reward += reward
        states[i + 1, :] = env.state
        actions[i, :] = env.action(action)
        scores[i, :] = np.array([reward, episode_reward, env.is_stable(new_state), env.is_contained(new_state)])
        state = new_state
        if done:
            break
        i += 1
    return states, actions, scores


def nSim(flag, agent, env, n):
    n_states = np.zeros((env.steps, env.observation_space.shape[0] - 6, n))
    n_actions = np.zeros((env.steps, env.action_space.shape[0], n))
    n_scores = np.zeros((env.steps, 4, n))
    bar = Bar('Processing', max=n)
    for k in range(n):
        bar.next()
        states, actions, scores = sim(flag, agent, env)
        n_states[:, :, k] = states
        n_actions[:, :, k] = actions
        n_scores[:, :, k] = scores
    return n_states, n_actions, n_scores


def nSim3D(n, agent, env, path, show=False):
    flag = True
    n_states, _, _ = nSim(flag, agent, env, n)
    plot_nSim3D(n_states, path, show)


def plot_nSim3D(n_states, path, show=False):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    n = n_states.shape[-1]
    for k in range(n):
        states = n_states[:, :, k]
        X = states[:, 3]
        Y = states[:, 4]
        Z = states[:, 5]
        ax.plot(X, Y, Z, '.b', alpha=0.8, markersize=1)

    fig.suptitle(r'# of flights = ' + '{} '.format(n), fontsize=20)
    ax.plot(0, 0, 0, '.r', alpha=1, markersize=1)
    if show:
        plt.show()
    else:
        fig.set_size_inches(33., 21.)
        plt.savefig(path + '/n_flights.png', dpi=300)

def plot_nSim2D(array3D, columns, time, show=True, file_name=None):
    steps, var, samples = array3D.shape
    index = pd.MultiIndex.from_product([range(samples), range(steps)], names=['samples', 'steps'])
    data = pd.DataFrame(data=array3D.reshape(steps * samples,var), index=index, columns=columns)
    _, axes = plt.subplots(6, 2)
    for sample, df in data.groupby(level=0):
        df['time'] = time
        if sample == 0:
            legend = True
        else:
            legend = False

        df.plot(x='time', subplots=True, ax=axes, legend=legend)
    
    if show:
        plt.show()
    else:
        plt.savefig(file_name, dpi=300)
