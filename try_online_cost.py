import numpy as np
import pathlib
import time
from GPS.utils import ContinuousDynamics, FiniteDiffCost, OnlineCost
from Linear.equations import f, W0
from dynamics import VEL_MIN, VEL_MAX
from env import QuadcopterEnv
from GPS.controller import iLQG, OnlineController
from simulation import plot_rollouts, rollout, n_rollouts
from matplotlib import pyplot as plt
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES  # , SCORE_NAMES
from utils import date_as_path
from dynamics import penalty, terminal_penalty
from ilqr import RecedingHorizonController
from tqdm import tqdm
# import pandas as pd

PATH = 'results_mpc/' + date_as_path() + '/'
pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)

env = QuadcopterEnv(u0=W0)  # AgentEnv(QuadcopterEnv())
# env.set_time(400, 16)
n_u = len(env.action_space.sample())
n_x = len(env.observation_space.sample())

# env.noise_on = False
dt = env.time[-1] - env.time[-2]
dynamics = ContinuousDynamics(
    f, n_x=n_x, n_u=n_u, u0=W0, dt=dt, method='lsoda')

# cost = FiniteDiffCostBounded(cost=penalty,
#                              l_terminal=terminal_penalty,
#                              state_size=n_x,
#                              action_size=n_u,
#                              u_bound=0.6 * W0
#                              )
cost = FiniteDiffCost(l=penalty,
                      l_terminal=terminal_penalty,
                      state_size=n_x,
                      action_size=n_u
                      )
N = env.steps - 1
low = env.action_space.low
high = env.action_space.high
offline_control = iLQG(dynamics, cost, N, low, high)
# 'results_offline/23_02_01_13_30/'
offline_control.load('results_offline/23_02_09_17_21/', 'ilqr_control.npz')

online_cost = OnlineCost(n_x, n_u, offline_control, nu=0.0, lamb=np.zeros(n_u))
online_control = OnlineController(dynamics, online_cost, N, low, high)
x0 = env.observation_space.sample()  # states_init[0]
agent = RecedingHorizonController(x0, online_control)
# expert = LinearAgent(env)

horizon = 50
EPISODES = 1


steps = env.steps - 1
# x0 = np.zeros(n_x)
# print(x0)
# np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(steps, n_u))
# us_init = np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(
#     N, n_u))
#
# us_init = np.zeros((steps, n_u))
_, us_init, _ = rollout(offline_control, env, state_init=x0)
# us_init = np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(
#    N, n_u))
states = np.zeros((EPISODES, env.steps + 1, len(env.state)))
actions = np.zeros((EPISODES, env.steps, env.action_space.shape[0]))
costs = list()  # np.zeros((EPISODES, env.steps - 1))


t1 = time.time()
traj = agent.control(us_init, step_size=horizon,
                     initial_n_iterations=50,
                     subsequent_n_iterations=25)

j = 0
C = np.empty_like(offline_control._C)
K = np.empty_like(offline_control._K)
k = np.empty_like(offline_control._k)
alpha = np.empty_like(offline_control._nominal_us)
for i in tqdm(range(offline_control.N // horizon)):
    xs, us = next(traj)
    C[i:i+horizon] = online_control._C[:horizon]
    K[i:i+horizon] = online_control._K[:horizon]
    alpha[i:i+horizon] = online_control.alpha
    k[i:i+horizon] = online_control._k[:horizon]
    states[:, j:j + horizon + 1] = xs
    actions[:, j:j + horizon] = us
    j += horizon

t2 = time.time()

print(f'Timpo de ejecución: {t2-t1}')


print('ya acabo el ajuste del mpc')

create_animation(states[:, :-1], actions[:, :-1], env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path=PATH + 'sample_rollouts/')


fig1, _ = plot_rollouts(states[:, :-1], env.time, STATE_NAMES, alpha=0.5)
fig1.savefig(PATH + 'state_rollouts.png')
fig2, _ = plot_rollouts(actions[:, :-1], env.time, ACTION_NAMES, alpha=0.5)
fig2.savefig(PATH + 'action_rollouts.png')


# fig4.savefig(PATH + 'eigvalues_C.png')

agent._controller.save(PATH)
print('los parametros del control fueron guardadados')
