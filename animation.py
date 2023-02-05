import numpy as np
import imageio
from Linear.equations import angles2rotation
from Linear.agent import LinearAgent
from GPS.controller import iLQG
from simulation import n_rollouts
from env import QuadcopterEnv
from numpy import cos, sin
from matplotlib import pyplot as plt
import os
from DDPG.utils import AgentEnv
from DDPG.ddpg import DDPGagent
from dynamics import transform_x, inv_transform_x
from simulation import plot_rollouts
from params import STATE_NAMES, ACTION_NAMES

PATH = 'test/'


def square(vec=np.zeros(3), R=np.identity(3)):
    r = np.linspace(-1, 1, 100)
    v1 = R @ np.array([.4, .4, 0])
    v2 = R @ np.array([.4, -.4, 0])
    q = np.array(list(map(lambda x: vec + x * v1.T, r)))
    p = np.array(list(map(lambda x: vec + x * v2.T, r)))
    # q = vec + r * v1.T
    # p = vec - r * v2.T
    return q.T, p.T


def apply_R(p, R):
    q = np.zeros_like(p)
    for i in range(p.shape[1]):
        q[:, i] = R@p[:, i]

    return q


def create_animation(states, actions, time, scores=None, state_labels=None,
                     action_labels=None, score_labels=None,
                     file_name='animation', path=PATH):
    '''
    Argumentos
    ----------
    states : `np.ndarray`
        Representa la trayectoria de estados. Es un arreglo con
        dimensiones (# trayectorias, # pasos, dimensión de estado).
    actions : `np.ndarray`
        Representa la trayectoria de acciones. Es un arreglo con
        dimensiones (# trayectorias, # pasos, dimensión de acción).
    time : `np.ndarray`
        Es un arreglo que representa el tiempo con
        dimensión (# pasos,).
    scores (opcional): `np.ndarray`
        Representa la trayectoria de puntajes. Es un arreglo con
        dimensiones (# trayectorias, # pasos, dimensión de puntaje).
    state_labels (opcional): `list`
        Lista de los nombres de los estados.
    action_labels (opcional): `list`
        Lista de los nombres de las acciones.
    score_labels (opcional): `list`
        Lista de los nombres de los puntajes.
    file_name (opcional): `str`
        Nombre del archivo. <file_name>.gif
    path (opcional): `str`
        Nombre de la carpeta donde se guardan las imagenes temporales.
    '''

    plt.style.use("fivethirtyeight")
    if len(states.shape) == 2:
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)
        if isinstance(scores, np.ndarray):
            scores = np.expand_dims(scores, axis=0)
    samples = actions.shape[0]
    steps = actions.shape[1]
    _scores = None
    file_name = path + file_name + '_{}.gif'
    for j in range(samples):
        if isinstance(scores, np.ndarray):
            _scores = scores[j]
        _create_frames(states[j], actions[j], time, scores=_scores,
                       state_labels=state_labels,
                       action_labels=action_labels,
                       score_labels=score_labels,
                       path=path, j=j)
        with imageio.get_writer(file_name.format(j), mode='i') as writer:
            for i in range(0, steps):
                image = imageio.v2.imread(path + f'/frame_{j}_{i}.png')
                writer.append_data(image)

                os.system('rm ' + path + f'frame_{j}_{i}.png')


def _create_frames(states: np.ndarray, actions: np.ndarray, time: np.ndarray,
                   scores=None, state_labels=None, action_labels=None,
                   score_labels=None, goal=None, path=PATH, j=None):
    '''
    Argumentos
    ----------
    states : `np.ndarray`
        Representa la trayectoria de estados. Arreglo con
        dimensiones (# pasos, dimensión de estado).
    actions : `np.ndarray`
        Representa la trayectoria de acciones. Arreglo con
        dimensiones (# pasos, dimensión de acción).
    time : `np.ndarray`
        Es un arreglo que representa el tiempo con
        dimensión (# pasos,).
    scores : `np.ndarray`
        Representa la trayectoria de los puntajes o información adicional.
        Arreglo con dimensiones (# pasos, dimensión de puntaje).
    goal : `np.ndarray`
        Representa el estado parcial objetivo. Ej. [x, y, z].
        Puede ser un arreglo con dimensiones
        (steps, dimensión de estado parcial) o (dimensión de estado parcial,).

    Referencias
    -----------
    1. https://towardsdatascience.com/create-panel-figure-layouts-in-matplotlib-with-gridspec-7ec79c218df0
    '''
    steps, action_dim = actions.shape
    state_dim = states.shape[-1]
    height_ratios = [1] * action_dim
    height_ratios += [1.5, 1.5]
    max_states = np.apply_along_axis(np.max, 0, states)
    min_states = np.apply_along_axis(np.min, 0, states)
    # max_actions = np.apply_along_axis(np.max, 0, actions)
    # min_actions = np.apply_along_axis(np.min, 0, actions)
    state_bounds = np.vstack([min_states, max_states])
    # action_bounds = np.vstack([min_actions, max_actions])
    scores_dim = None
    if isinstance(scores, np.ndarray):
        scores_dim = scores.shape[-1]
    if not isinstance(state_labels, list):
        state_labels = ['$s_{}$'.format(i) for i in range(1, state_dim+1)]
    if not isinstance(action_labels, list):
        action_labels = ['$a_{}$'.format(i) for i in range(1, action_dim + 1)]
    for i in range(steps):
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(nrows=action_dim + 2, ncols=2,
                              height_ratios=height_ratios, width_ratios=[1, 3])
        axes_action = np.array([fig.add_subplot(gs[i, 0])
                               for i in range(action_dim)])
        plot_rollouts(actions[:i+1], time, action_labels, axes_action)
        ax1 = fig.add_subplot(gs[-2, 1])
        ax2 = fig.add_subplot(gs[-1, 1])
        plot_rollouts(states[:i+1, 3:6], time[:i+1],
                      state_labels[3:6], ax=ax1, subplots=False)
        plot_rollouts(states[:i+1, 9:12], time[:i+1],
                      state_labels[9:12], ax=ax2, subplots=False,
                      colors=['darkorange'])
        ax_3d = fig.add_subplot(gs[0:4, 1], projection='3d')
        _quadcopter_frame(states[:i+1], goal, state_bounds, ax=ax_3d)
        x, y, z = states[i, 3:6]
        ax_3d.set_title('$t=$ {:.2f}'.format(
            time[i]) + '\n $x=$ {:.2f}, $y=$ {:.2f}, $z=$ {:.2f}'.format(x, y, z), fontsize=15)
        if isinstance(scores, np.ndarray):
            ax_scores = fig.add_subplot(gs[-2:, 0])
            ax_scores.axis([0, 10, 0, 2 * scores_dim])
            for k in range(scores_dim):
                ax_scores.text(
                    0, 2 * k, score_labels[k] +
                    ' ={:.2f}'.format(scores[i, k]),
                    fontsize=25)
            ax_scores.set_axis_off()
        if isinstance(j, int):
            file_name = path + f'frame_{j}_{i}.png'
        else:
            file_name = path + f'frame_{i}.png'
        plt.savefig(file_name)
        plt.close()


def _quadcopter_frame(states, goal_pos=None, state_bounds=None, ax=None):
    state_dim = states.shape[-1]
    u, v, w, x, y, z, p, q, r, psi, theta, phi = np.split(
        states, state_dim, axis=1)
    xmin, ymin, zmin = state_bounds[0, 3:6]
    xmax, ymax, zmax = state_bounds[1, 3:6]
    if not isinstance(ax, plt.Axes):
        ax = plt.axes(projection='3d')
    if isinstance(goal_pos, np.ndarray) | isinstance(goal_pos, list):
        ax.plot(goal_pos[0], goal_pos[1], goal_pos[2], 'r.')
    ax.plot(x, y, z, alpha=0.5, linestyle='-.')
    R = angles2rotation(
        np.array([psi[-1], theta[-1], phi[-1]]).flatten(), flatten=False)
    p, q = square(np.array([x[-1], y[-1], z[-1]]).flatten(), R=R)
    xp, yp, zp = p  # p_[0], p_[1], p_[2]
    xq, yq, zq = q  # q_[0], q_[1], q_[2]
    ax.plot(xp, yp, zp, 'k')
    ax.plot(xq, yq, zq, 'k')
    ax.plot(x[-1], y[-1], z[-1], 'bo', linewidth=0.2)
    ax.set_xlim3d(xmin - 1, xmax + 1)
    ax.set_ylim3d(ymin - 1, ymax + 1)
    ax.set_zlim3d(zmin - 1, zmax + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


if __name__ == '__main__':
    # t = np.arange(0, 40, .1)
    # size = len(t)
    # state = f(t)
    # angles = g(t, tmax=t[-1], tmin=t[0])
    env = QuadcopterEnv()
    env.set_time(305, 15)
    # env4agent = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    agent = LinearAgent(env)
    # agent = iLQRAgent()
    # agent = DDPGagent(env4agent)
    # agent.load('results_ddpg/12_9_112/')
    states, actions, scores = n_rollouts(agent, env, n=5)
    # states = np.apply_along_axis(inv_transform_x, -1, states)
    # pos = states[:, 3:6]
    # angles = states[:, 9:]
    size = len(env.time)
    score_names = ['$r_t$', r'$\sum^T r_t$']
    create_animation(states, actions, env.time, scores=scores,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     score_labels=score_names,
                     file_name='flight',
                     path='Linear/sample_rollouts/')
