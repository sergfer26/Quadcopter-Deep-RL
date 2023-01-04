import time
import pathlib
import send_email
import numpy as np
from matplotlib import pyplot as plt
from Linear.equations import W0, f
from GPS.controller import iLQRAgent, MPC
from GPS.utils import OfflineCost, MPCCost
from GPS.utils import ContinuousDynamics
from env import QuadcopterEnv
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from Linear.agent import LinearAgent
from simulation import plot_rollouts, rollout
from get_report import create_report
from animation import create_animation
from utils import date_as_path
from multiprocessing import Process
from dynamics import penalty, teminal_penalty


def fit_lqg(env, expert, path='', i=None):
    dt = env.time[-1] - env.time[-2]
    low = env.observation_space.low
    high = env.observation_space.high
    n_u = 4  # env.num_actions
    n_x = 12  # env.num_states
    steps = env.steps - 1
    dynamics = ContinuousDynamics(
        f, state_size=n_x, action_size=n_u, u0=W0, dt=dt, method='lsoda')

    ####### Instancias control iLQG #######
    cost = OfflineCost(penalty, teminal_penalty,
                       state_size=n_x,
                       action_size=n_u,
                       eta=10,
                       _lambda=np.zeros(n_u),
                       nu=0,
                       N=steps
                       )

    control = iLQRAgent(dynamics, cost, steps, low, high,
                        state_names=STATE_NAMES)
    xs_init, us_init, _ = rollout(expert, env)
    control.fit_control(xs_init[0], us_init)
    if isinstance(i, int):
        file_name = f'control_{i}.npz'
    else:
        file_name = f'control.npz'
    control.save(path=path, file_name=file_name)


def fit_mpc_control(env, path='', i=None, j=None, N=374, horizon=25):
    dt = env.time[-1] - env.time[-2]
    low = env.observation_space.low
    high = env.observation_space.high
    n_u = 4  # env.num_actions
    n_x = 12  # env.num_states
    dynamics = ContinuousDynamics(
        f, state_size=n_x, action_size=n_u, u0=W0, dt=dt, method='lsoda')

    ####### Instancias control MPC-iLQG #######
    mpc_cost = MPCCost(n_x, n_u, nu=1, _lambda=np.zeros(n_u), N=N)

    mpc_control = MPC(dynamics, mpc_cost, N, low, high,
                      horizon=horizon, state_names=STATE_NAMES)

    if isinstance(i, int):
        file_path = path + f'control_{i}.npz'
    else:
        file_path = path + 'control.npz'

    mpc_control.update_offline_control(file_path=file_path)
    xs_init = mpc_control.off_nominal_xs
    us_init = mpc_control.off_nominal_us
    mpc_control.control(xs_init, us_init)

    if isinstance(i, int) and isinstance(j, int):
        file_name = f'mpc_control_{i}_{j}.npz'
    else:
        file_name = 'mpc_control.npz'

    mpc_control.save(path, file_name)


def main(path):
    N = 10
    M = 4
    horizon = 10
    env = QuadcopterEnv()
    n_u = len(env.action_space.sample())
    n_x = len(env.observation_space.sample())

    dt = env.time[-1] - env.time[-2]
    T = env.steps - 1

    ####### Instancias control lineal #######

    expert = LinearAgent(env)
    time_max = env.time_max
    env.set_time(T + horizon, time_max + int(horizon*dt))

    ti = time.time()
    processes_lqg = list()
    for i in range(N):
        p = Process(target=fit_lqg, args=(env, expert, path, i))
        processes_lqg.append(p)
        p.start()

    ti = time.time()
    for p in processes_lqg:
        p.join()
    tf = time.time()
    print(f'tiempo de ajuste iLQG: {tf - ti}')

    env.set_time(T, time_max)

    processes_mpc = list()
    for i, j in zip(range(N), range(M)):
        p = Process(target=fit_mpc_control, args=(
            env, path, i, j, T, horizon))
        processes_mpc.append(p)
        p.start()

    ti = time.time()
    for p in processes_mpc:
        p.join()
    tf = time.time()
    print(f'tiempo de ajuste MPC: {tf - ti}')

    states = np.empty((N, M, T + 1, n_x))
    actions = np.empty((N, M, T, n_u))
    for i, j in range(N, M):
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

    fig1, _ = plot_rollouts(states[:, :-1], env.time, STATE_NAMES)
    fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES)
    fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES)
    fig1.savefig(path + 'state_rollouts.png')
    fig2.savefig(path + 'action_rollouts.png')
    fig3.savefig(path + 'score_rollouts.png')
    create_report(path,
                  title='Prueba MPC',
                  method='gcl',
                  extra_method='ilqr'
                  )

    subpath = path + 'mpc_rollouts/'
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
    print('Termino de simualcion...')
    create_animation(states, actions, env.time, scores=scores,
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
