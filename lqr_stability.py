import pathlib
from params import STATE_NAMES
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
from dynamics import f, penalty, terminal_penalty
from utils import date_as_path
from params import state_space


def rollout4mp(agent, env, mp_list, n=1, states_init=None):
    '''
    states : (n, env.steps, env.observation_space.shape[0])
    '''
    states = n_rollouts(agent, env, n=n, states_init=states_init)[0]
    mp_list.append(states)


def rollouts(agent, env, sims, state_space, x0=None, num_workers=None,
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

    if not isinstance(x0, np.ndarray):
        x0 = np.zeros_like(state_space[0, 0])

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

    states = np.array(list(states))  # (6, sims, T, n_x)
    # Ordenar indices
    aux = list(map(lambda i: (np.nonzero(states[i, 0, 0])[
               0][0], i), list(range(num_workers))))
    aux.sort(key=lambda x: x[0])
    indices = [x for _, x in aux]
    states = states[indices]
    if callable(inv_transform_x):
        states = np.apply_along_axis(inv_transform_x, -1, states)

    return states


def confidence_region(states, goal_states=None, c=1e-1):
    if not isinstance(goal_states, np.ndarray):
        goal_states = np.zeros_like(states)
    return np.apply_along_axis(lambda x: np.linalg.norm(x-goal_states) <= c,
                               -1, states)


if __name__ == '__main__':
    plt.style.use("fivethirtyeight")
    env = QuadcopterEnv()
    control_path = 'models/'
    PATH = 'results_ilqr/stability_analysis/'+date_as_path()+'/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    sims = int(1e2)
    eps = 4e-1

    labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
              ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
              ('$r$', '$\psi$')
              ]

    # 1. Setup
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    dynamics = ContinuousDynamics(f, n_x, n_u, dt=env.dt)
    cost = FiniteDiffCost(penalty, terminal_penalty, n_x, n_u)

    agent = DummyController(control_path, f'ilqr_control_{env.steps}.npz')
    states = rollouts(agent, env, sims, state_space)

    bool_state = np.apply_along_axis(
        lambda x: np.linalg.norm(x) < eps, -1, states[:, :, -1])
    fig, axes = plt.subplots(
        figsize=(12, 10), nrows=state_space.shape[1]//3, ncols=3, dpi=250)
    axs = axes.flatten()
    mask1 = np.apply_along_axis(lambda x, y: np.greater(
        abs(x), y), -1, states[:, 0, 0], 0)
    mask2 = state_space[1] > 0
    init_states = states[:, :, 0]
    sc = list()
    for i in range(init_states.shape[0]):
        mask = abs(init_states[i, 0]) > 0
        label = np.array(STATE_NAMES)[mask]
        aux = plot_classifier(init_states[i, :, mask],
                              bool_state[i], x_label=label[0],
                              y_label=label[1], ax=axs[i],
                              )[1]
        sc.append(aux)

    fig.suptitle(f'Control iLQR \\ $\eps=${eps}')
    fig.savefig(PATH + 'stability_region.png')

    np.savez(
        PATH + 'stability_region.npz',
        states=states[:, :, [0, env.steps]],
        bounds=state_space[1]
    )

    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones de control: ' + PATH,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={env.steps} \n time_max={env.time_max} \n sims={sims}',
               path2images=PATH
               )
