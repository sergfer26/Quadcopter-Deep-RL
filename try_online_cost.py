import numpy as np
import pathlib
import time
from GPS.utils import ContinuousDynamics, FiniteDiffCost, OnlineCost
from Linear.equations import f, W0
from env import QuadcopterEnv
from GPS.controller import iLQG, OnlineController, OnlineMPC
from simulation import plot_rollouts
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES
from utils import date_as_path
from dynamics import penalty, terminal_penalty
# from ilqr import RecedingHorizonController
from tqdm import tqdm

PATH = 'results_online/' + date_as_path() + '/'
pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)

env = QuadcopterEnv(u0=W0)  # AgentEnv(QuadcopterEnv())
# env.set_time(400, 16)
n_u = len(env.action_space.sample())
n_x = len(env.observation_space.sample())

# env.noise_on = False
dt = env.time[-1] - env.time[-2]
dynamics = ContinuousDynamics(
    f, n_x=n_x, n_u=n_u, u0=W0, dt=dt, method='lsoda')

cost = FiniteDiffCost(l=penalty,
                      l_terminal=terminal_penalty,
                      state_size=n_x,
                      action_size=n_u
                      )
N = env.steps - 1
offline_control = iLQG(dynamics, cost, N)
# 'results_offline/23_02_01_13_30/'
offline_control.load('results_offline/23_02_16_13_50/', 'ilqr_control.npz')

online_cost = OnlineCost(n_x, n_u, offline_control,
                         nu=0.01, lamb=np.zeros((N, n_u)))
online_control = OnlineController(dynamics, online_cost, N)
x0 = env.observation_space.sample()  # states_init[0]
# RecedingHorizonController(x0, online_control)
agent = OnlineMPC(x0, online_control)
# expert = LinearAgent(env)

horizon = 50

xs, us = offline_control.rollout(x0)
states = np.zeros((env.steps + 1, len(env.state)))
actions = np.zeros((env.steps, env.action_space.shape[0]))
costs = list()  # np.zeros((EPISODES, env.steps - 1))


t1 = time.time()
agent.x0 = x0
traj = agent.control(us, step_size=horizon,
                     initial_n_iterations=50,
                     subsequent_n_iterations=50)

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
    states[j:j + horizon + 1] = xs
    actions[j:j + horizon] = us
    j += horizon

t2 = time.time()

print(f'Timpo de ejecución: {t2-t1}')


print('ya acabo el ajuste del mpc')

fig1, _ = plot_rollouts(states[:-1], env.time, STATE_NAMES, alpha=0.5)
fig1.savefig(PATH + 'state_rollouts.png')
fig2, _ = plot_rollouts(actions[:-1], env.time, ACTION_NAMES, alpha=0.5)
fig2.savefig(PATH + 'action_rollouts.png')


create_animation(states[:-1], actions[:-1], env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path=PATH + 'sample_rollouts/')

agent._controller.save(PATH)
print('los parametros del control fueron guardadados')
