import argparse
import os
import pathlib
import imageio
import numpy as np
import matplotlib as mpl
from Linear import LinearAgent
from Linear.equations import angles2rotation
from simulation import n_rollouts
from env import QuadcopterEnv
# from numpy import cos, sin
from matplotlib import pyplot as plt
from simulation import plot_rollouts
from params import (
    ACTION_NAMES,
    PARAMS_DDPG,
    PARAMS_TRAIN_DDPG,
    REWARD_NAMES,
    STATE_NAMES,
)
from policy import Policy
from DDPG.ddpg import DDPGagent
from DDPG.utils import AgentEnv
from GPS.controller import DummyController
from dynamics import transform_x, inv_transform_x


PATH = 'test/'
DEFAULT_GPS_PATH = PARAMS_TRAIN_DDPG['behavior_path'].rstrip('/')
DEFAULT_ILQR_PATH = 'models'


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
                     action_labels=None, score_labels=None, goal=None,
                     title=None, file_name='animation', path=PATH, 
                     delete_frames=True, style='fivethirtyeight',
                     show_scores=True):
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
    delete_frames (opcional): `bool`
        Si es `True`, borra las imágenes temporales después de crear el gif.
    style (opcional): `str`
        Estilo de matplotlib usado para crear la animación.
    show_scores (opcional): `bool`
        Si es `True`, muestra los puntajes durante la animación.
    '''

    plt.style.use(style)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    if len(states.shape) == 2:
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)
        if isinstance(scores, np.ndarray):
            scores = np.expand_dims(scores, axis=0)
    samples = actions.shape[0]
    steps = actions.shape[1]
    _scores = None
    gif_path = os.path.join(path, file_name + '_{}.gif')
    for j in range(samples):
        if show_scores and isinstance(scores, np.ndarray):
            _scores = scores[j]
        _create_frames(states[j], actions[j], time, scores=_scores,
                       state_labels=state_labels,
                       action_labels=action_labels,
                       score_labels=score_labels,
                       goal=goal,
                       path=path, j=j, title=title)
        with imageio.get_writer(gif_path.format(j), mode='i') as writer:
            for i in range(0, steps):
                frame_path = os.path.join(path, f'frame_{j}_{i}.png')
                image = imageio.v2.imread(frame_path)
                writer.append_data(image)

                if delete_frames:
                    os.remove(frame_path)


def _create_frames(states: np.ndarray, actions: np.ndarray, time: np.ndarray,
                   scores=None, state_labels=None, action_labels=None,
                   score_labels=None, goal=None, path=PATH, j=None,
                   title=None, fontsize=30):
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
                      state_labels[3:6], axes=ax1, subplots=False)
        plot_rollouts(states[:i+1, 9:12], time[:i+1],
                      state_labels[9:12], axes=ax2, subplots=False,
                      colors=['darkorange'])
        ax_3d = fig.add_subplot(gs[0:4, 1], projection='3d')
        _quadcopter_frame(states[:i+1], goal, state_bounds, ax=ax_3d)
        x, y, z = states[i, 3:6]
        ax_3d.set_title('$t=$ {:.2f}'.format(
            time[i]) +
            '\n $x=$ {:.2f}, $y=$ {:.2f}, $z=$ {:.2f}'.format(x, y, z),
            fontsize=15)
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
            file_name = os.path.join(path, f'frame_{j}_{i}.png')
        else:
            file_name = os.path.join(path, f'frame_{i}.png')

        if isinstance(title, str):
            fig.suptitle(title, fontsize=fontsize)
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
        ax.plot(goal_pos[0], goal_pos[1], goal_pos[2], 'r.', alpha=0.1)
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


def _build_agent(agent_type: str, env: QuadcopterEnv, agent_path: str = None):
    if agent_type == 'linear':
        return LinearAgent(env), env

    if agent_type == 'ilqr':
        control_path = agent_path or DEFAULT_ILQR_PATH
        file_name = f'ilqr_control_{env.steps}.npz'
        if os.path.isdir(control_path):
            control_path = control_path.rstrip('/') + '/'
        return DummyController(control_path, file_name), env

    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    other_env.noise_on = False

    if agent_type == 'gps':
        policy_path = agent_path or DEFAULT_GPS_PATH
        policy = Policy(other_env, PARAMS_DDPG['hidden_sizes'])
        policy.load(policy_path)
        return policy, other_env

    if agent_type == 'ddpg':
        if not isinstance(agent_path, str):
            raise ValueError(
                'A checkpoint directory must be provided with --agent-path '
                'when agent type is ddpg.'
            )
        agent = DDPGagent(
            other_env,
            hidden_sizes=PARAMS_DDPG['hidden_sizes'],
            actor_learning_rate=eval(PARAMS_DDPG['actor_learning_rate']),
            critic_learning_rate=PARAMS_DDPG['critic_learning_rate'],
            gamma=PARAMS_DDPG['gamma'],
            tau=PARAMS_DDPG['tau'],
            max_memory_size=PARAMS_DDPG['max_memory_size'],
        )
        agent.load(agent_path.rstrip('/') + '/')
        return agent, other_env

    raise ValueError(f'Unknown agent type: {agent_type}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=750,
                        help='Number of simulation steps passed to env.set_time.')
    parser.add_argument('--dt', type=float, default=0.04,
                        help='Time step passed to env.set_time.')
    parser.add_argument('--agent', type=str,
                        choices=['linear', 'ilqr', 'ddpg', 'gps'],
                        default='linear',
                        help='Type of agent used to generate rollouts.')
    parser.add_argument('--agent-path', type=str, default=None,
                        help='Optional directory used to load gps/ddpg/ilqr agents.')
    parser.add_argument('--file-name', type=str, default='flight',
                        help='Base name of the generated gif files.')
    parser.add_argument('--style', type=str, default='fivethirtyeight',
                        help='Argument forwarded to plt.style.use.')
    parser.add_argument('--show-scores', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Show or hide score text in animation frames.')
    parser.add_argument('--n-rollouts', type=int, default=2,
                        help='Number of rollouts used to build the animation.')
    parser.add_argument('--path', type=str, default='Linear/sample_rollouts/',
                        help='Directory where frames and gifs are written.')
    parser.add_argument('--delete-frames',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Delete temporary PNG frames after building gifs.')
    args = parser.parse_args()

    env = QuadcopterEnv()
    env.set_time(args.steps, args.dt)
    agent, rollout_env = _build_agent(args.agent, env, args.agent_path)
    transform_states = inv_transform_x if args.agent in {'gps', 'ddpg'} else None
    states, actions, scores = n_rollouts(
        agent, rollout_env, n=args.n_rollouts, t_x=transform_states
    )

    create_animation(states, actions, env.time,
                     scores=scores if args.show_scores else None,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     score_labels=REWARD_NAMES,
                     file_name=args.file_name,
                     path=args.path,
                     delete_frames=args.delete_frames,
                     style=args.style,
                     show_scores=args.show_scores
                     )
