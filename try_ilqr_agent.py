import numpy as np
import pathlib
from GPS.utils import ContinuousDynamics, FiniteDiffCost
from Linear.equations import f, W0
from dynamics import VEL_MIN, VEL_MAX
from env import QuadcopterEnv
from GPS.controller import iLQRAgent
from simulation import plot_rollouts, rollout, n_rollouts
from matplotlib import pyplot as plt
from Linear.agent import LinearAgent
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES  # , SCORE_NAMES
from get_report import create_report
from utils import date_as_path
from dynamics import transform_x, transform_u
from GPS.utils import FiniteDiffCostBounded
# import pandas as pd

PATH = 'results_ilqr/' + date_as_path() + '/'
pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)

env = QuadcopterEnv(u0=W0)  # AgentEnv(QuadcopterEnv())
env.set_time(400, 16)
n_u = len(env.action_space.sample())
n_x = len(env.observation_space.sample())

# env.noise_on = False
dt = env.time[-1] - env.time[-2]
dynamics = ContinuousDynamics(
    f, state_size=n_x, action_size=n_u, u0=W0, dt=dt, method='lsoda')

# cost = FiniteDiffCostBounded(cost=lambda x, u, i: - env.get_reward(x, u),
#                              l_terminal=lambda x, i: -
#                              env.get_reward(x, np.zeros(n_u)),
#                              state_size=n_x,
#                              action_size=n_u,
#                              u_bound=0.6 * W0
#                              )
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

x0 = states_init[0]
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

plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.plot(costs[-1])
# ax.plot(costs[0], color='green', marker='o', linestyle='dashed')
ax.set_title('Costo')
fig.savefig(PATH + 'train_performance.png')
# plt.show()

create_animation(states, actions, env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path=PATH + 'sample_rollouts/')

agent.reset()
states, actions, scores = n_rollouts(agent, env, n=100)

fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
fig1.savefig(PATH + 'state_rollouts.png')
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
fig2.savefig(PATH + 'action_rollouts.png')
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
fig3.savefig(PATH + 'score_rollouts.png')

create_report(PATH, 'Ajuste iLQR', method=None, extra_method='ilqr')
agent.save(PATH)
print('los parametros del control fueron guardadados')

sample_indices = np.random.randint(states.shape[0], size=3)
states_samples = states[sample_indices]
actions_samples = actions[sample_indices]
scores_samples = scores[sample_indices]

# states = np.apply_along_axis(transform_x, -1, states)
# actions = np.apply_along_axis(transform_u, -1, actions)
# new_states = states[:, 1:, :]
# states = states[:, :-1, :]
# dones = np.zeros((actions.shape[0], env.steps - 1), dtype=bool)
# dones[:, -1] = True
# kwargs = dict(states=states, actions=actions,
#               next_states=new_states, dones=dones)
# np.savez(PATH + 'memory_transformed_x_u.npz', **kwargs)


create_animation(states_samples, actions_samples, env.time,
                 scores=scores_samples,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 score_labels=REWARD_NAMES,
                 file_name='flight',
                 path=PATH + 'sample_rollouts/')
