

# Crear la lista especial del main
# Hacer una fución extendida que reciba la lista especial y los parametros de rollout
# Llamar estas dos partes

import time
import glob
import os
import pathlib
from GPS.policy import Policy
from params import PARAMS_DDPG
from DDPG.utils import AgentEnv
from gym import spaces
from multiprocessing import Process
import multiprocessing as mp
import numpy as np
from simulation import n_rollouts
from GPS.controller import DummyController
from GPS.utils import ContinuousDynamics
from env import QuadcopterEnv
from utils import date_as_path, plot_classifier
from send_email import send_email
from matplotlib import pyplot as plt
from ilqr.cost import FiniteDiffCost
from dynamics import f, inv_transform_x, transform_x, penalty, terminal_penalty


def rollout4mp(agent, env, mp_list, n=1, states_init=None):
    states = n_rollouts(agent, env, n=n, states_init=states_init)[0]
    mp_list.append(states[:, -1])


def rollouts(agent, env, sims, state_space, num_workers=None,
             inv_transform_x=None, transform_x=None):
    if not isinstance(num_workers, int):
        num_workers = state_space.shape[1]

    final_states = mp.Manager().list()
    process_list = list()
    if hasattr(agent, 'env'):
        other_env = agent.env
    else:
        other_env = env
    init_states = np.empty((num_workers, sims, env.state.shape[0]))
    for i in range(num_workers):
        env.observation_space = spaces.Box(
            low=state_space[0, i], high=state_space[1, i], dtype=np.float64
        )
        init_states[i] = np.array(
            [env.observation_space.sample() for _ in range(sims)])
        init_state = init_states[i]
        if callable(transform_x):
            init_state = np.apply_along_axis(transform_x, -1, init_state)
        p = Process(target=rollout4mp, args=(
            agent, other_env, final_states, sims, init_state)
        )
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    final_states = np.array(list(final_states))
    if callable(inv_transform_x):
        final_states = np.apply_along_axis(inv_transform_x, -1, final_states)

    return init_states, final_states


def classifier(state, goal_state=None, c=1e-1):
    if not isinstance(goal_state, np.ndarray):
        goal_state = np.zeros_like(state)
    return np.apply_along_axis(criterion, 0, state, goal_state, c=c).all()


def criterion(x, y=0, c=1e-1):
    return abs(x - y) < c


def confidence_region(states, goal_states=None, c=1e-1):
    if not isinstance(goal_states, np.ndarray):
        goal_states = np.zeros_like(states)
    return np.apply_along_axis(classifier, -1, states, goal_states, c)


def get_color(bools):
    return np.array(['b' if b else 'r' for b in bools])


if __name__ == '__main__':
    plt.style.use("fivethirtyeight")
    PATH = 'results_sims/' + date_as_path() + '/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    results_path = 'results_gps/23_04_13_14_57/buffer/'
    path = PATH

    sims = int(1e1)

    labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
              ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
              ('$r$', '$\psi$')
              ]

    # 1. Setup
    env = QuadcopterEnv()
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    other_env.noise_on = False
    hidden_sizes = PARAMS_DDPG['hidden_sizes']
    policy = Policy(other_env, hidden_sizes)
    policy.load('results_gps/23_04_13_14_57/')
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    dynamics = ContinuousDynamics(f, n_x, n_u, dt=env.dt)
    cost = FiniteDiffCost(penalty, terminal_penalty, n_x, n_u)
    high = np.array([
        # u, v, w, x, y, z, p, q, r, psi, theta, phi
        [.1, 0., 0., .1, 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., .1, 0., 0., .1, 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., .1, 0., 0., .1, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., .001, 0., 0., 0., 0., np.pi/64],
        [0., 0., 0., 0., 0., 0., 0., .001, 0., 0., np.pi/64, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., .001, np.pi/64, 0., 0.]
    ])

    low = -high

    state_space = np.stack([low, high])

    # 2. iLQR controls' simulations
    filelist = glob.glob(results_path + 'control_*')
    n_files = len(filelist)
    N = np.load(filelist[0])['K'].shape[0]
    env.set_time(N, env.dt)
    for k in range(n_files):
        agent = DummyController(results_path, f'control_{k}.npz')
        init_states, final_states = rollouts(agent, env, sims, state_space)

        bool_state = confidence_region(final_states)

        cluster = np.apply_along_axis(get_color, -1, bool_state)
        fig, axes = plt.subplots(
            figsize=(14, 10), nrows=high.shape[0]//3, ncols=3, dpi=250,
            sharey=True)
        axs = axes.flatten()
        for label, i in zip(labels, range(high.shape[0])):
            states = init_states[i, :, high[i] > 0]
            plot_classifier(states, cluster[i], x_label=label[0],
                            y_label=label[1], ax=axs[i])

        fig.suptitle(f'Control {k}')
        fig.savefig(path + f'samples_control_{k}.png')

    # 3. Policy's simulations
    init_states, final_states = rollouts(policy, env, sims, state_space,
                                         inv_transform_x=inv_transform_x,
                                         transform_x=transform_x)
    bool_state = confidence_region(final_states)

    cluster = np.apply_along_axis(get_color, -1, bool_state)
    labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
              ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
              ('$r$', '$\psi$')
              ]
    fig, axes = plt.subplots(
        figsize=(14, 10), nrows=high.shape[0]//3, ncols=3, dpi=250,
        sharey=True)
    axs = axes.flatten()
    for label, i in zip(labels, range(high.shape[0])):
        states = init_states[i, :, high[i] > 0]
        plot_classifier(
            states, cluster[i], x_label=label[0], y_label=label[1], ax=axs[i])

    fig.suptitle('Política')
    fig.savefig(path + 'samples_policy.png')

    np.savez(
        PATH + 'states.npz',
        init=init_states,
        final=final_states,
        high=high
    )

    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones',
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={N} \n time_max={env.time_max} \n sims={sims} \n path: '+results_path,
               path2images=path
               )

    # fig, _ = plot_rollouts(np.array(init_states),
    #                        np.arange(0, 100, 1), STATE_NAMES)
    # fig.suptitle('Estados iniciales')
    # plt.show()
    # fig, _ = plot_rollouts(np.array(final_states),
    #                        np.arange(0, 100, 1), STATE_NAMES)
    # fig.suptitle('Estados finales')
    # plt.show()
