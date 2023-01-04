import numpy as np
import pathlib
from matplotlib import pyplot as plt
from Linear.equations import W0, f
from GPS.controller import iLQRAgent, MPC
from GPS.utils import OfflineCost, MPCCost
from GPS.utils import ContinuousDynamics
from env import QuadcopterEnv
from params import STATE_NAMES, ACTION_NAMES
from Linear.agent import LinearAgent
from simulation import plot_rollouts
from animation import create_animation
from utils import date_as_path

env = QuadcopterEnv(u0=W0)  # AgentEnv(QuadcopterEnv())
n_u = len(env.action_space.sample())
n_x = len(env.observation_space.sample())

PATH = 'results_mpc/' + date_as_path() + '/'
pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)

# env.noise_on = False
dt = env.time[-1] - env.time[-2]
dynamics = ContinuousDynamics(
    f, state_size=n_x, action_size=n_u, u0=W0, dt=dt, method='lsoda')

N = 25
cost = OfflineCost(lambda x, u, i: - env.get_reward(x, u),
                   l_terminal=lambda x, i: -
                   env.get_reward(x, np.zeros(n_u)),
                   state_size=n_x,
                   action_size=n_u,
                   eta=1000,
                   _lambda=np.zeros(n_u),
                   nu=0,
                   N=N
                   )

low = env.observation_space.low
high = env.observation_space.high
agent = iLQRAgent(dynamics, cost, N, low, high,
                  state_names=STATE_NAMES)


horizon = 4
env.set_time(N, 1)
cost_mpc = MPCCost(n_x, n_u, nu=1, _lambda=np.zeros(n_u), N=N)
mpc_agent = MPC(dynamics, cost_mpc, N, low, high,
                horizon=horizon, state_names=STATE_NAMES)

# x0 = np.zeros(n_x)
# print(x0)
# np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(steps, n_u))
# states_init, us_init, scores_init = rollout(expert, env, state_init=x0)
# us_init = np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(
#    N, n_u))

# x0 = states_init[0]
# xs, us, cost_trace = agent.fit_control(x0, us_init)
agent.load('results_ilqr/22_12_29_22_33/')

xs = agent._nominal_xs
us = agent._nominal_us
mpc_agent.update_offline_control(agent)
xs, us = mpc_agent.control(xs, us, initial_n_iterations=10)

plot_rollouts(xs, env.time, STATE_NAMES)
plt.show()
plot_rollouts(us, env.time, ACTION_NAMES)
plt.show()
create_animation(xs, us, env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='flight',
                 path=PATH + 'sample_rollouts/')
