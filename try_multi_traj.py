import time
import pathlib
import send_email
import numpy as np
from functools import partial
from ilqr.cost import FiniteDiffCost
from Linear.equations import W0, f
from GPS.controller import iLQRAgent, DummyControl
from GPS.utils import OfflineCost, MPCCost, FiniteDiffCostBounded
from GPS.utils import ContinuousDynamics
from env import QuadcopterEnv
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from Linear.agent import LinearAgent
from simulation import plot_rollouts, rollout
from get_report import create_report
# from animation import create_animation
from utils import date_as_path
from multiprocessing import Process, Pool
from dynamics import penalty, terminal_penalty
from GPS.params import PARAMS_iLQR as params
from multiprocessing import Manager

cost_kwargs = dict(l=penalty,
                   l_terminal=terminal_penalty,
                   state_size=n_x,
                   action_size=n_u,
                   eta=10,
                   lamb=np.zeros(n_u),
                   nu=0,
                   N=steps)

dynamics_kwargs = dict(f=f,
                       n_x=n_x,
                       n_u=n_u,
                       u0=W0,
                       dt=dt,
                       method='lsoda'
                       )


def fit_ilqg(env, expert, cost_kwargs, dynamics_kwargs, i, T, M, path=''):
    '''
    i : int
        Indice de trayectoria producida por iLQG.
    T : int
        Número de pasos de la trayectoria.
    horizon : int
        Tamaño de la ventana de horizonte del MPC.
    M : int
        Indice de trayectoria producida por MPC.
    '''
    low = env.observation_space.low
    high = env.observation_space.high
    steps = env.steps - 1
    dynamics = ContinuousDynamics(**dynamics_kwargs)

    # ###### Instancias control iLQG #######
    cost = OfflineCost(**cost_kwargs)
    control = iLQRAgent(dynamics, cost, steps, low, high)
    xs_init, us_init, _ = rollout(expert, env)
    control.fit_control(xs_init[0], us_init)
    control.save(path=path, file_name=f'control_{i}.npz')
    with Pool(processes=M) as pool:
        pool.map(partial(_fit_child, env,
                 T, path, i), range(M))
        pool.close()
        pool.join()


def _fit_child(env, T, path, i, j):

    file_name = f'control_{i}.npz'
    control = DummyControl(T, path, file_name)
    xs, us, _ = rollout(control, env)
    np.savez(path + f'traj_{i}_{j}.npz',
             xs=xs,
             us=us
             )


def main(path):
    N = 3
    M = 2
    env = QuadcopterEnv()

    T = env.steps - 1

    # ###### Instancias control lineal #######

    expert = LinearAgent(env)

    processes = list()
    for i in range(N):
        p = Process(target=fit_mpc, args=(
            env, expert, i, T, M, path))
        processes.append(p)
        p.start()

    ti = time.time()
    for p in processes:
        p.join()

    tf = time.time()
    print(f'tiempo de ajuste de trayectorias MPC: {tf - ti}')

    breakpoint()

    return path


if __name__ == '__main__':
    PATH = 'borrame/' + date_as_path() + '/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    main(PATH)
