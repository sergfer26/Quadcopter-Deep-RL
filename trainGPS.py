import pathlib
import torch
import numpy as np
from tqdm import tqdm
from utils import date_as_path
import matplotlib.pyplot as plt
from Linear.equations import f, W0
from GPS import GPS, Policy, iLQRAgent, ContinuousDynamics
from ilqr.cost import FiniteDiffCost
from env import QuadcopterEnv
from Linear.agent import LinearAgent
from simulation import rollout
from DDPG.utils import AgentEnv
from dynamics import transform_x, transform_u
from dynamics import inv_transform_u, inv_transform_x
from dynamics import penalty, terminal_penalty


def train_gps(gps: GPS, K, path):
    losses = np.empty(K)
    with tqdm(total=K) as pbar:
        for k in range(K):
            pbar.set_description(f'Update {k + 1}/'+str(K))
            loss, _ = gps.update_policy(path)
            losses[k] = loss
            pbar.set_postfix(loss='{:.2f}'.format(loss))
            pbar.update(1)
    return losses


if __name__ == '__main__':
    # 1. Setup
    PATH = 'results_gps/' + date_as_path() + '/'
    pathlib.Path(PATH + 'buffer/').mkdir(parents=True, exist_ok=True)
    env = QuadcopterEnv()
    dt = env.time[-1] - env.time[-2]
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    policy = Policy(other_env, [64, 64])
    policy.load_state_dict(torch.load(
        'results_ddpg/12_9_113/actor', map_location='cpu'))
    dynamics_kwargs = dict(f=f, n_x=n_x, n_u=n_u, dt=dt, u0=W0, method='lsoda')

    # 2. Pre-training
    # dynamics = ContinuousDynamics(**dynamics_kwargs)
    # cost = FiniteDiffCost(penalty, terminal_penalty, n_x, n_u)
    # low = env.observation_space.low
    # high = env.observation_space.high
    # control = iLQRAgent(dynamics, cost, env.steps - 1, low, high)
    # expert = LinearAgent(env)
    # xs, us_init, _ = rollout(expert, env, state_init=np.zeros(n_x))
    # control.fit_control(xs[0], us_init)
    # control.save(PATH + 'buffer/', 'control.npz')

    # 3. Training
    gps = GPS(env,
              policy,
              dynamics_kwargs,
              penalty,
              terminal_penalty,
              t_x=transform_x,
              t_u=transform_u,
              inv_t_x=inv_transform_x,
              inv_t_u=inv_transform_u,
              u_bound=0.6 * W0
              )
    losses = train_gps(gps, 2, PATH)

    breakpoint()
