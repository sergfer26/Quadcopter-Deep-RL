

# Crear la lista especial del main
# Hacer una fución extendida que reciba la lista especial y los parametros de rollout
# Llamar estas dos partes

import time
import pathlib
from GPS.policy import Policy
from params import PARAMS_DDPG
from DDPG.utils import AgentEnv
from gym import spaces
from multiprocessing import Process
import multiprocessing as mp
import numpy as np
from simulation import n_rollouts
from Linear.agent import LinearAgent
from env import QuadcopterEnv
from utils import date_as_path
from send_email import send_email
# from Linear.classifier import postion_vs_velocity
from simulation import plot_rollouts
from params import STATE_NAMES
from matplotlib import pyplot as plt
from dynamics import inv_transform_x, transform_x


def rollout4mp(agent, env, mp_list, n=1, states_init=None):
    states = n_rollouts(agent, env, n=n, states_init=states_init)[0]
    mp_list.append(states[:, -1])


# def foo(lista):
#    lista.append(np.random.randint(100))


if __name__ == '__main__':
    PATH = 'results_sims/' + date_as_path() + '/'
    pathlib.Path(PATH + 'buffer/').mkdir(parents=True, exist_ok=True)
    flag = True
    sims = 5000
    n_process = 6
    final_states = mp.Manager().list()
    otra_lista = list()
    env = QuadcopterEnv()
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    other_env.noise_on = False
    hidden_sizes = PARAMS_DDPG['hidden_sizes']
    policy = Policy(other_env, hidden_sizes)
    policy.load('results_gps/23_04_13_14_57/')
    init_states = np.empty((n_process, sims, env.state.shape[0]))
    high = np.array([
        # u, v, w, x, y, z, p, q, r, psi, theta, phi
        [1., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 10., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., .1, 0., 0., 0., 0., np.pi/16],
        [0., 0., 0., 0., 0., 0., 0., .1, 0., 0., np.pi/16, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., .1, np.pi/16, 0., 0.]
    ])

    for i in range(n_process):
        env.observation_space = spaces.Box(
            low=-high[i], high=high[i], dtype=np.float64
        )
        init_states[i] = np.array(
            [env.observation_space.sample() for _ in range(sims)])

        p = Process(target=rollout4mp, args=(
            policy, other_env, final_states, sims,
            np.apply_along_axis(transform_x, -1, init_states[i])
        )
        )
        otra_lista.append(p)
        p.start()

    ti = time.time()
    for p in otra_lista:
        p.join()
    tf = time.time()
    print('tiempo de simulación: ', tf - ti)
    final_states = np.array(list(final_states))
    # final_states = final_states.reshape(
    #     final_states.shape[0] * final_states.shape[1], final_states.shape[-1])

    np.savez(
        PATH + 'states.npz',
        init=init_states,
        final=final_states,
        high=high
    )
    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones',
               reciever='sfernandezm97@ciencias.unam.mx',
               message='Listo!'
               )

    # fig, _ = plot_rollouts(np.array(init_states),
    #                        np.arange(0, 100, 1), STATE_NAMES)
    # fig.suptitle('Estados iniciales')
    # plt.show()
    # fig, _ = plot_rollouts(np.array(final_states),
    #                        np.arange(0, 100, 1), STATE_NAMES)
    # fig.suptitle('Estados finales')
    # plt.show()
