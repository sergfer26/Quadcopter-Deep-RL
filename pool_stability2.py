import time
import pathlib
from stability import stability
from multiprocessing import Process
from utils import date_as_path
from send_email import send_email
from GPS.controller import DummyController
from params import STATE_NAMES, state_space
from GPS.controller import iLQG
from GPS.utils import ContinuousDynamics, FiniteDiffCost
from env import QuadcopterEnv
from simulation import rollout
from Linear.equations import f, W0
from dynamics import penalty, terminal_penalty

if __name__ == '__main__':
    path = 'models/'
    save_path = 'results_ilqr/multi_stability/'+date_as_path()+'/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    processes = list()
    sims = int(1e4)
    eps = 4e-1
    with_x0 = True
    env = QuadcopterEnv()
    n_u = len(env.action_space.sample())
    n_x = len(env.observation_space.sample())
    expert = DummyController(path, f'ilqr_control_{env.steps}.npz')
    cost = FiniteDiffCost(l=penalty,
                          l_terminal=terminal_penalty,
                          state_size=n_x,
                          action_size=n_u
                          )
    dt = env.dt
    N = env.steps
    dynamics = ContinuousDynamics(
        f, n_x=n_x, n_u=n_u, u0=W0, dt=dt)  # ContinuousDynamics

    for i in range(7):
        save_name = f'stability_{i}'
        agent = iLQG(dynamics, cost, N)
        us_init = rollout(expert, env, state_init=x0)[1]
        _ = agent.fit_control(x0, us_init)

        p = Process(target=stability, args=(agent,
                                            state_space,
                                            STATE_NAMES,
                                            save_path,
                                            save_name,
                                            eps,
                                            with_x0,
                                            sims,
                                            ))
        processes.append(p)
        p.start()

    ti = time.time()
    for p in processes:
        p.join()

    tf = time.time()
    total_t = tf - ti
    send_email(credentials_path='credentials.txt',
               subject='Termino de simulaciones de control: ' + save_path,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'tiempo total: {total_t}',
               path2images=save_path
               )
