import numpy as np
from ILQR.utils import ContinuousDynamics, FiniteDiffCost
from Linear.equations import f, W0
from dynamics import VEL_MIN, VEL_MAX
from env import QuadcopterEnv
from ILQR.agent import iLQRAgent
from simulation import plot_rollouts, rollout, n_rollouts
from matplotlib import pyplot as plt
from Linear.agent import LinearAgent
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES  # , SCORE_NAMES
# import pandas as pd

env = QuadcopterEnv(u0=W0)  # AgentEnv(QuadcopterEnv())
n_u = len(env.action_space.sample())
n_x = len(env.observation_space.sample())

# env.noise_on = False
dt = env.time[-1] - env.time[-2]
dynamics = ContinuousDynamics(
    f, state_size=n_x, action_size=n_u, u0=W0, dt=dt, method='lsoda')
cost = FiniteDiffCost(l=lambda x, u, i: -env.get_reward(x, u),
                      l_terminal=lambda x, i: -
                      env.get_reward(x, np.zeros(n_u)),
                      state_size=n_x,
                      action_size=n_u
                      )
N = env.steps - 1
low = env.observation_space.low
high = env.observation_space.high
agent = iLQRAgent(dynamics, cost, env.steps-1, low, high,
                  state_names=STATE_NAMES)
expert = LinearAgent(env)


EPISODES = 1


steps = env.steps - 1
x0 = np.zeros(n_x)
# print(x0)
# np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(steps, n_u))
states_init, us_init, scores_init = rollout(expert, env, state_init=x0)
# us_init = np.random.uniform(low=VEL_MIN, high=VEL_MAX, size=(
#    N, n_u))

# x0 = states_init[0]
states = np.zeros((EPISODES, env.steps, len(env.state)))
actions = np.zeros((EPISODES, env.steps - 1, env.action_space.shape[0]))
costs = list()  # np.zeros((EPISODES, env.steps - 1))
for ep in range(EPISODES):
    # np.zeros(n_x)
    # x0 = env.observation_space.sample()
    xs, us, cost_trace = agent.fit_control(x0, us_init)
    us_init = us
    states[ep] = xs
    actions[ep] = us
    costs.append(cost_trace)

print('ya acabo el ajuste del control')
# states = np.apply_along_axis(, states)
# plot_nSim2D(states, env.time, STATE_NAMES)
# plt.show()
#
#
# plot_nSim2D(actions, env.time, ACTION_NAMES)
# plt.show()

# fig, ax = plt.subplots(figsize=(2, 2), dpi=200)
# ax.plot(costs[-1])
# ax.plot(costs[0], color='green', marker='o', linestyle='dashed')
# ax.set_title('Costo')
# plt.show()

create_animation(states, actions, env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path='ILQR/sample_rollouts/')

agent.reset()
states, actions, scores = n_rollouts(agent, env, n=2)
create_animation(states, actions, env.time,
                 scores=scores,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 score_labels=REWARD_NAMES,
                 file_name='flight',
                 path='ILQR/sample_rollouts/')
