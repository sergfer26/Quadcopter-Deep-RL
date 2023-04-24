import time
import pathlib
import send_email
import numpy as np
from functools import partial
from Linear.equations import W0, f
import matplotlib.pyplot as plt
from ilqr import RecedingHorizonController
from GPS.controller import OfflineController, OnlineController
from GPS.utils import OfflineCost, OnlineCost
from GPS.utils import ContinuousDynamics
from env import QuadcopterEnv
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from Linear.agent import LinearAgent
from simulation import plot_rollouts, rollout
from get_report import create_report
from animation import create_animation
from utils import date_as_path
from multiprocessing import Process, Pool
from dynamics import penalty, terminal_penalty
from GPS.params import PARAMS_OFFLINE, PARAMS_ONLINE


def fit_mpc(env, expert, i, T, horizon, M, path=''):
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
    dt = env.dt
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    steps = env.steps
    dynamics = ContinuousDynamics(
        f, n_x=n_x, n_u=n_u, u0=W0, dt=dt)

    # ###### Instancias control iLQG #######
    cost = OfflineCost(penalty, terminal_penalty,
                       n_x=n_x,
                       n_u=n_u,
                       eta=5,
                       lamb=np.zeros((steps, n_u)),
                       nu=np.zeros(T),
                       T=steps
                       )
    cost.update_control(
        file_path='results_offline/23_02_09_17_21/ilqr_control.npz'
    )
    # 'results_offline/23_02_01_13_30/ilqr_control.npz'
    control = OfflineController(dynamics, cost, steps)
    xs_init, us_init, _ = rollout(expert, env, state_init=np.zeros(n_x))
    control.x0 = xs_init[0]
    control.us_init = us_init
    # fit_control(xs_init[0], us_init)
    control.optimize(kl_step=PARAMS_OFFLINE['kl_step'], min_eta=cost.eta)
    control.save(path=path, file_name=f'control_{i}.npz')
    x0 = [env.observation_space.sample() for _ in range(M)]
    # _fit_child(x0, low_action, high_action, dt, T, horizon, path, i, 0)
    with Pool(processes=M) as pool:
        pool.map(partial(_fit_child, x0, dt, T, horizon, path, i), range(M))
        pool.close()
        pool.join()


def _fit_child(x0, dt, T, horizon, path, i, j):
    if isinstance(x0, list):
        x0 = x0[j]
    n_u = W0.shape[-1]
    n_x = x0.shape[-1]
    dynamics = ContinuousDynamics(
        f, n_x=n_x, n_u=n_u, u0=W0, dt=dt)
    control = OfflineController(dynamics, None, T)
    control.load(path, file_name=f'control_{i}.npz')

    # ###### Instancias control MPC-iLQG #######
    cost = OnlineCost(n_x, n_u, control, nu=np.zeros(T),
                      lamb=np.zeros((T, n_u)), F=PARAMS_ONLINE['F'])

    mpc_control = OnlineController(dynamics, cost, T)
    agent = RecedingHorizonController(x0, mpc_control)

    _, us_init = control.rollout(x0)

    traj = agent.control(us_init,
                         step_size=horizon,
                         initial_n_iterations=50,
                         subsequent_n_iterations=25)
    states = np.empty((T + 1, n_x))
    actions = np.empty((T, n_u))
    C = np.empty_like(control._C)
    K = np.empty_like(control._K)
    k = np.empty_like(control._k)
    alpha = np.empty_like(control._nominal_us)
    r = 0
    for t in range(mpc_control.N // horizon):
        xs, us = next(traj)
        C[t: t + horizon] = mpc_control._C[:horizon]
        K[t: t + horizon] = mpc_control._K[:horizon]
        alpha[t: t + horizon] = mpc_control.alpha
        k[t: t + horizon] = mpc_control._k[:horizon]
        states[r: r + horizon + 1] = xs
        actions[r: r + horizon] = us
        r += horizon

    file_name = f'mpc_control_{i}_{j}.npz'
    mpc_control._C = C
    mpc_control._K = K
    mpc_control._k = k
    mpc_control.alpha = alpha
    mpc_control._nominal_us = actions
    mpc_control._nominal_xs = states
    mpc_control.save(path, file_name)


def main(path):
    N = 6
    M = 4
    horizon = PARAMS_ONLINE['step_size']
    env = QuadcopterEnv()
    n_u = len(env.action_space.sample())
    n_x = len(env.observation_space.sample())
    time_max = env.time_max

    T = env.steps

    # ###### Instancias control lineal #######

    expert = LinearAgent(env)
    processes = list()
    for i in range(N):
        p = Process(target=fit_mpc, args=(env, expert, i, T, horizon, M, path))
        processes.append(p)
        p.start()

    ti = time.time()
    for p in processes:
        p.join()

    tf = time.time()
    print(f'tiempo de ajuste de trayectorias MPC: {tf - ti}')

    states = np.empty((N, M, T + 1, n_x))
    actions = np.empty((N, M, T, n_u))
    for i in range(N):
        for j in range(M):
            file = np.load(path + f'mpc_control_{i}_{j}.npz')
            states[i, j] = file['xs']
            actions[i, j] = file['us']

    states = states.reshape((N * M, T + 1, -1))
    actions = actions.reshape((N * M, T, -1))
    scores = np.empty((N * M, T, 2))
    scores[:, :, 0] = np.apply_along_axis(
        env.get_reward, -1, states[:, :-1], actions)
    scores[:, :, 1] = np.cumsum(scores[:, :, 0], axis=-1)

    env.set_time(T, time_max)

    plt.style.use("fivethirtyeight")
    fig1, _ = plot_rollouts(states[:, :-1], env.time, STATE_NAMES, alpha=0.1)
    fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.1)
    fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.1)
    fig1.savefig(path + 'state_rollouts.png')
    fig2.savefig(path + 'action_rollouts.png')
    fig3.savefig(path + 'score_rollouts.png')
    create_report(path,
                  title='Prueba MPC',
                  method=None,
                  extra_method='ilqr'
                  )

    subpath = path + 'mpc_rollouts/'
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
    print('Termino de simualcion...')
    create_animation(states, actions, env.time,
                     scores=scores,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     score_labels=REWARD_NAMES,
                     path=subpath
                     )

    return path


if __name__ == '__main__':
    PATH = 'results_mpc/' + date_as_path() + '/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    send_email.report_sender(main, args=[PATH])
    print(PATH)
