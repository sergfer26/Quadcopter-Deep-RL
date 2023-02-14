import numpy as np
import pathlib
import time
from GPS.utils import ContinuousDynamics, FiniteDiffCost
from Linear.equations import f, W0
from dynamics import VEL_MIN, VEL_MAX
from env import QuadcopterEnv
from GPS.controller import iLQG
from simulation import plot_rollouts, rollout, n_rollouts
from matplotlib import pyplot as plt
from Linear.agent import LinearAgent
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES  # , SCORE_NAMES
from utils import date_as_path
from dynamics import penalty, terminal_penalty
from ilqr import RecedingHorizonController
from tqdm import tqdm
import warnings

# import pandas as pd
warnings.filterwarnings("ignore")
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

controller = iLQG(dynamics, cost, N)
controller.load('results_offline/23_02_09_17_21/', 'ilqr_control.npz')
x0 = env.observation_space.sample()  # states_init[0]
agent = RecedingHorizonController(x0, controller)
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
_, us_init, _ = rollout(controller, env, state_init=x0)
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
# for i in tqdm(range((controller.N // horizon))):
#     xs, us = next(traj)
#     C, K = controller._C[:horizon], controller._K[:horizon]
#     alpha, k = controller.alpha, controller._k[:horizon]
#     states[:, j:j + horizon + 1] = xs
#     actions[:, j:j + horizon] = us
#     j += horizon

C = np.empty_like(controller._C)
K = np.empty_like(controller._K)
k = np.empty_like(controller._k)
alpha = np.empty_like(controller._nominal_us)
for i in tqdm(range(controller.N // horizon)):
    xs, us = next(traj)
    C[i: i + horizon] = controller._C[:horizon]
    K[i: i + horizon] = controller._K[:horizon]
    alpha[i: i + horizon] = controller.alpha
    k[i: i + horizon] = controller._k[:horizon]
    states[:, j: j + horizon + 1] = xs
    actions[:, j: j + horizon] = us
    j += horizon


file_path = PATH + 'mpc_control.npz'
np.savez(file_path,
         C=np.round(C, 3),
         K=K,
         k=k,
         xs=states,
         us=actions,
         alpha=alpha
         )

t2 = time.time()

print(f'Tiempo de ejecución: {t2-t1}')


print('ya acabo el ajuste del mpc')

create_animation(states[:, :-1], actions[:, :-1], env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path=PATH + 'sample_rollouts/')


fig1, _ = plot_rollouts(states[:, :-1], env.time, STATE_NAMES, alpha=0.05)
fig1.savefig(PATH + 'state_rollouts.png')
fig2, _ = plot_rollouts(actions[:, :-1], env.time, ACTION_NAMES, alpha=0.05)
fig2.savefig(PATH + 'action_rollouts.png')


# fig4.savefig(PATH + 'eigvalues_C.png')

agent._controller.save(PATH)
print('los parametros del control fueron guardadados')
