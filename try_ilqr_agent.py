import numpy as np
import pathlib
from GPS.utils import ContinuousDynamics, FiniteDiffCost
from Linear.equations import f, W0
from env import QuadcopterEnv
from GPS.controller import iLQG
from simulation import plot_rollouts, rollout, n_rollouts
from matplotlib import pyplot as plt
from Linear.agent import LinearAgent
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES  # , SCORE_NAMES
from get_report import create_report
from utils import date_as_path
from dynamics import penalty, terminal_penalty
# import pandas as pd

PATH = 'results_ilqr/' + date_as_path() + '/'
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
low = env.action_space.low
high = env.action_space.high
agent = iLQG(dynamics, cost, N, low, high)
expert = LinearAgent(env)


EPISODES = 1


steps = env.steps - 1
x0 = np.zeros(n_x)

_, us_init, _ = rollout(expert, env, state_init=x0)
states = np.zeros((EPISODES, env.steps, len(env.state)))
actions = np.zeros((EPISODES, env.steps - 1, env.action_space.shape[0]))
costs = list()  # np.zeros((EPISODES, env.steps - 1))
for ep in range(EPISODES):
    xs, us, cost_trace = agent.fit_control(x0, us_init)
    us_init = us
    states[ep] = xs
    actions[ep] = us
    costs.append(cost_trace)

print('ya acabo el ajuste del control')


plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.plot(costs[-1])
ax.set_title('Costo')
fig.savefig(PATH + 'train_performance.png')

create_animation(states, actions, env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path=PATH + 'sample_rollouts/')

agent.reset()
states, actions, scores = n_rollouts(
    agent, env, n=100)

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

create_animation(states_samples, actions_samples, env.time,
                 scores=scores_samples,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 score_labels=REWARD_NAMES,
                 file_name='flight',
                 path=PATH + 'sample_rollouts/')
