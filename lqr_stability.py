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
            low=state_space[0, i]+x0, high=state_space[1, i]+x0,
            dtype=np.float64)
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
    aux = list(map(lambda i: (np.nonzero(states[i, 0, 0] - x0)[
               0][0], i), list(range(num_workers))))
    aux.sort(key=lambda x: x[0])
    indices = [x for _, x in aux]
    states = states[indices]
    if callable(inv_transform_x):
        states = np.apply_along_axis(inv_transform_x, -1, states)

    return states


def stability(path, file_name, save_path, save_name='stability_region',
              eps=4e-1, with_x0=False, sims=int(1e4)):
    plt.style.use("fivethirtyeight")
    agent = DummyController(path, file_name)
    env = QuadcopterEnv()
    x0 = np.zeros_like(env.state)
    if with_x0:
        x0 = agent._nominal_xs[0]
    states = rollouts(agent, env, sims, state_space, x0=x0)
    bool_state = np.apply_along_axis(
        lambda x: np.linalg.norm(x) < eps, -1, states[:, :, -1])
    fig, axes = plt.subplots(
        figsize=(14, 10), nrows=state_space.shape[1]//3, ncols=3, dpi=300)
    axs = axes.flatten()
    init_states = states[:, :, 0]
    # sc = list()
    for i in range(init_states.shape[0]):
        mask = abs(init_states[i, 0] - x0) > 0
        label = np.array(STATE_NAMES)[mask]
        plot_classifier(init_states[i, :, mask],
                        bool_state[i], x_label=label[0],
                        y_label=label[1], ax=axs[i],
                        )[1]
    #    sc.append(aux)

    fig.suptitle(f'Control iLQR \n $\epsilon=${eps}')
    fig.savefig(save_path + save_name + '.png')

    np.savez(
        save_path + save_name + '.npz',
        states=states[:, :, [0, env.steps]],
        bounds=state_space[1]
    )


if __name__ == '__main__':
    control_path = 'models/'
    PATH = 'results_ilqr/stability_analysis/'+date_as_path()+'/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    sims = int(1e1)
    eps = 4e-1
    T = 750
    stability(control_path, f'ilqr_control_{T}.npz', PATH, eps=eps, sims=sims)

    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones de control: ' + PATH,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={T} \n eps={eps} \n sims={sims}',
               path2images=PATH
               )
