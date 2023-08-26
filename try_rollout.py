import glob
import pathlib
from GPS.policy import Policy
from params import PARAMS_DDPG, STATE_NAMES
from DDPG.utils import AgentEnv
from gym import spaces
from multiprocessing import Process
import multiprocessing as mp
import numpy as np
from simulation import n_rollouts
from GPS.controller import DummyController
from GPS.utils import ContinuousDynamics
from env import QuadcopterEnv
from utils import plot_classifier
from send_email import send_email
from matplotlib import pyplot as plt
from ilqr.cost import FiniteDiffCost
from dynamics import f, inv_transform_x, transform_x, penalty, terminal_penalty
from utils import date_as_path


def rollout4mp(agent, env, mp_list, n=1, states_init=None):
    '''
    states : (n, env.steps, env.observation_space.shape[0])
    '''
    states = n_rollouts(agent, env, n=n, states_init=states_init)[0]
    mp_list.append(states)


def rollouts(agent, env, sims, state_space, num_workers=None,
             inv_transform_x=None, transform_x=None):
    '''
    Retorno
    --------
    states : (np.ndarray)
        dimensión -> (state_space.shape[0], sims, env.steps,
                        env.observation_space.shape[0])
    '''
    if not isinstance(num_workers, int):
        num_workers = state_space.shape[1]

    states = mp.Manager().list()
    process_list = list()
    if hasattr(agent, 'env'):
        other_env = agent.env
    else:
        other_env = env
    init_states = np.empty((num_workers, sims, env.state.shape[0]))
    for i in range(num_workers):
        env.observation_space = spaces.Box(
            low=state_space[0, i], high=state_space[1, i], dtype=np.float64)
        init_states[i] = np.array(
            [env.observation_space.sample() for _ in range(sims)])
        init_state = init_states[i]
        if callable(transform_x):
            init_state = np.apply_along_axis(transform_x, -1, init_state)
        p = Process(target=rollout4mp, args=(
            agent, other_env, states, sims, init_state
        )
        )
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    states = np.array(list(states))
    if callable(inv_transform_x):
        states = np.apply_along_axis(inv_transform_x, -1, states)

    return states


def classifier(state, goal_state=None, c=5e-1):
    if not isinstance(goal_state, np.ndarray):
        goal_state = np.zeros_like(state)
    return np.apply_along_axis(criterion, 0, state, goal_state, c=c).all()


def criterion(x, y=0, c=5e-1):
    return abs(x - y) < c


def confidence_region(states, goal_states=None, c=5e-1):
    if not isinstance(goal_states, np.ndarray):
        goal_states = np.zeros_like(states)
    return np.apply_along_axis(classifier, -1, states, goal_states, c)


def get_color(bools):
    return np.array(['b' if b else 'r' for b in bools])


if __name__ == '__main__':
    PATH = 'results_gps/23_07_31_12_15/'  # 'results_gps/23_04_13_14_57/'
    results_path = PATH + 'buffer/'
    policy_path = PATH + 'rollouts/' + date_as_path() + '/policy/'
    control_path = PATH + 'rollouts/' + date_as_path() + '/control/'
    pathlib.Path(policy_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(control_path).mkdir(parents=True, exist_ok=True)

    sims = int(1e1)

    labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
              ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
              ('$r$', '$\psi$')
              ]

    # 1. Setup
    env = QuadcopterEnv()
    T = int(60 / env.dt)
    list_steps = np.array([5, 10, 30, 60]) / env.dt
    time_max = env.time_max
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    other_env.noise_on = False
    hidden_sizes = PARAMS_DDPG['hidden_sizes']
    policy = Policy(other_env, hidden_sizes)
    policy.load(PATH)
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    dynamics = ContinuousDynamics(f, n_x, n_u, dt=env.dt)
    cost = FiniteDiffCost(penalty, terminal_penalty, n_x, n_u)
    high = np.array([
        # u, v, w, x, y, z, p, q, r, psi, theta, phi
        [10., 0., 0., 20., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 10., 0., 0., 20., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 10., 0., 0., 20., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., np.pi/2],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., np.pi/2, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., np.pi/2, 0., 0.]
    ])

    low = -high

    state_space = np.stack([low, high])

    env.set_time(T, env.dt)
    # 3. Policy's simulations
    states = rollouts(policy, env, sims, state_space,
                      inv_transform_x=inv_transform_x,
                      transform_x=transform_x)
    mask1 = np.apply_along_axis(lambda x, y: np.greater(
        abs(x), y), -1, states[:, 0, 0], 0)
    mask2 = high > 0
    indices = np.array([np.where(np.all(mask1 == mask2[i], axis=1))[0]
                        for i in range(6)]).squeeze()
    states = states[indices]
    init_states = states[:, :, 0]
    # steps=int(t * env.dt))
    for t in list_steps:
        bool_state = confidence_region(states[:, :, int(t)])
        cluster = np.apply_along_axis(get_color, -1, bool_state)
        fig, axes = plt.subplots(figsize=(14, 10), nrows=len(labels)//3,
                                 ncols=3, dpi=250, sharey=True)
        axs = axes.flatten()
        for i in range(init_states.shape[0]):
            mask = abs(init_states[i, 0]) > 0
            label = np.array(STATE_NAMES)[mask]
            plot_classifier(
                init_states[i, :, mask],
                cluster[i], x_label=label[0],
                y_label=label[1],
                ax=axs[i])
        fig.suptitle(f'Política, tiempo: {t * env.dt}')
        fig.savefig(policy_path + f'samples_policy_{int(t * env.dt)}.png')

    np.savez(
        policy_path + f'states_{int(env.time_max)}.npz',
        states=states,
        high=high
    )

    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones de política: ' + policy_path,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={env.steps} \n time_max={env.dt * T} \n sims={sims}',
               path2images=policy_path
               )

    # 2. iLQR controls' simulations
    filelist = glob.glob(results_path + 'control_*')
    n_files = len(filelist)
    N = np.load(filelist[0])['K'].shape[0]
    env.set_time(N, env.dt)
    for k in range(n_files):
        agent = DummyController(results_path, f'control_{k}.npz')
        states = rollouts(agent, env, sims, state_space,
                          init_states)

        bool_state = confidence_region(states[:, :, -1])

        cluster = np.apply_along_axis(get_color, -1, bool_state)
        fig, axes = plt.subplots(
            figsize=(14, 10), nrows=high.shape[0]//3, ncols=3, dpi=250,
            sharey=True)
        axs = axes.flatten()
        mask1 = np.apply_along_axis(lambda x, y: np.greater(
            abs(x), y), -1, states[:, 0, 0], 0)
        mask2 = high > 0
        indices = np.array([np.where(np.all(mask1 == mask2[i], axis=1))[
            0] for i in range(6)]).squeeze()
        states = states[indices]
        init_states = states[:, :, 0]
        for i in range(init_states.shape[0]):
            mask = abs(init_states[i, 0]) > 0
            label = np.array(STATE_NAMES)[mask]
            plot_classifier(init_states[i, :, mask],
                            cluster[i], x_label=label[0],
                            y_label=label[1], ax=axs[i]
                            )
        np.savez(
            control_path + f'states_{k}.npz',
            states=states
        )

        fig.suptitle(f'Control {k}')
        fig.savefig(control_path + f'samples_control_{k}.png')

    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones de control: ' + control_path,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={env.steps} \n time_max={env.time_max} \n sims={sims}',
               path2images=control_path
               )
