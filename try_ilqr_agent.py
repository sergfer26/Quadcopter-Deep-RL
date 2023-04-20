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
    f, n_x=n_x, n_u=n_u, u0=W0, dt=dt)  # ContinuousDynamics

cost = FiniteDiffCost(l=penalty,
                      l_terminal=terminal_penalty,
                      state_size=n_x,
                      action_size=n_u
                      )

N = env.steps - 1
agent = iLQG(dynamics, cost, N)
expert = LinearAgent(env)


steps = env.steps - 1
x0 = np.zeros(n_x)

us_init = rollout(expert, env, state_init=x0)[1]
# us_init = np.apply_along_axis(
#     constrain, -1, us_init, env.action_space.low, env.action_space.high)
costs = list()  # np.zeros((EPISODES, env.steps - 1))
xs, us, cost_trace = agent.fit_control(x0, us_init)
print('ya acabo el ajuste del control')
costs.append(cost_trace)
agent.save(PATH, f'ilqr_control_{int(N)}.npz')
print('los parametros del control fueron guardadados')

plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.plot(costs[-1])
ax.set_title('Costo')
fig.savefig(PATH + 'train_performance.png')

create_animation(xs, us, env.time,
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

sample_indices = np.random.randint(states.shape[0], size=2)
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
