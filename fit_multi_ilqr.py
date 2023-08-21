import numpy as np
import pathlib
import time
from GPS.utils import ContinuousDynamics, FiniteDiffCost
from Linear.equations import f, W0
from env import QuadcopterEnv
from GPS.controller import iLQG
from simulation import rollout
from Linear.agent import LinearAgent
from params import STATE_NAMES, state_space  # , SCORE_NAMES
from utils import date_as_path
from dynamics import penalty, terminal_penalty
from multiprocessing import Process
from lqr_stability import stability
from send_email import send_email
from GPS.controller import DummyController


def get_agent(env):
    n_u = len(env.action_space.sample())
    n_x = len(env.observation_space.sample())
    dt = env.dt
    dynamics = ContinuousDynamics(
        f, n_x=n_x, n_u=n_u, u0=W0, dt=dt)  # ContinuousDynamics

    cost = FiniteDiffCost(l=penalty,
                          l_terminal=terminal_penalty,
                          state_size=n_x,
                          action_size=n_u
                          )

    N = env.steps
    return iLQG(dynamics, cost, N)


def fit_agent(agent, x0, us_init, i=0, path=''):
    _ = agent.fit_control(x0, us_init)
    agent.save(path, f'control_{i}.npz')


if __name__ == '__main__':
    trajs = 6
    eps = 4e-1
    sims = int(1e4)
    PATH = 'results_ilqr/' + date_as_path() + '/'
    pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)
    env = QuadcopterEnv(u0=W0)
    expert = LinearAgent(env)

    # 1. Setup agents
    agents = [get_agent(env) for _ in range(trajs + 1)]
    x0 = np.zeros(agents[0].num_states)
    us_init = rollout(expert, env, state_init=x0)[1]

    # 2. Fit agents
    for i in range(0, trajs + 1):
        if i > 0:
            x0 = env.observation_space.sample()
            us_init = rollout(agents[0], env, state_init=x0)[1]
        fit_agent(agents[i], x0, us_init, i=0, path=PATH)

    # 3. Analize stability of every agent
    processes = list()
    with_x0 = True
    for i in range(0, trajs + 1):
        save_name = f'stability_{i}'
        agent = DummyController(PATH, f'control_{i}.npz')
        p = Process(target=stability, args=(agent,
                                            state_space,
                                            STATE_NAMES,
                                            PATH,
                                            save_name,
                                            eps,
                                            with_x0,
                                            sims,
                                            )
                    )
        processes.append(p)
        p.start()

    ti = time.time()
    for p in processes:
        p.join()

    tf = time.time()
    total_t = tf - ti
    print('tiempo total: ', total_t)

    send_email(credentials_path='credentials.txt',
               subject='Termino de analisis de estabilidad: ' + PATH,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={total_t} \n eps={eps} \n sims={sims}',
               path2images=PATH
               )
