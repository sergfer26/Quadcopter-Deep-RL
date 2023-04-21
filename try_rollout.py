

# Crear la lista especial del main
# Hacer una fución extendida que reciba la lista especial y los parametros de rollout
# Llamar estas dos partes

import time
from multiprocessing import Process
from threading import Thread
import multiprocessing as mp
import numpy as np
from simulation import rollout, n_rollouts
from Linear.agent import LinearAgent
from env import QuadcopterEnv
# from Linear.classifier import postion_vs_velocity
from simulation import plot_rollouts
from params import STATE_NAMES
from matplotlib import pyplot as plt


def rollout4mp(agent, env, mp_list, n=1, state_init=None):
    states = n_rollouts(agent, env, n, state_init=state_init)[0]
    mp_list.append(states[:, -1])


# def foo(lista):
#    lista.append(np.random.randint(100))


if __name__ == '__main__':
    flag = True
    final_states = mp.Manager().list()
    otra_lista = list()
    env = QuadcopterEnv()
    agent = LinearAgent(env)
    init_states = list()

    for _ in range(5):
        init_state = env.observation_space.sample()
        init_states.append(init_state)
        p = Process(target=rollout4mp, args=(
            agent, env, final_states, init_state))
        otra_lista.append(p)
        p.start()
    print('for rápido')

    ti = time.time()
    for p in otra_lista:
        p.join()
    tf = time.time()
    print(tf - ti)
    breakpoint()

    # fig, _ = plot_rollouts(np.array(init_states),
    #                        np.arange(0, 100, 1), STATE_NAMES)
    # fig.suptitle('Estados iniciales')
    # plt.show()
    # fig, _ = plot_rollouts(np.array(final_states),
    #                        np.arange(0, 100, 1), STATE_NAMES)
    # fig.suptitle('Estados finales')
    # plt.show()
