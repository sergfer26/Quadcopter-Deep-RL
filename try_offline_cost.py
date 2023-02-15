import numpy as np
import pathlib
import scipy
from GPS.utils import ContinuousDynamics
from Linear.equations import f, W0
from env import QuadcopterEnv
from GPS.controller import OfflineController
from simulation import plot_rollouts, rollout, n_rollouts
from matplotlib import pyplot as plt
from Linear.agent import LinearAgent
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from get_report import create_report
from utils import date_as_path
from dynamics import penalty, terminal_penalty
from GPS.utils import OfflineCost
from GPS.params import PARAMS_OFFLINE as PARAMS
# import pandas as pd

OLD_PATH = 'results_ilqr/23_02_08_22_01/'  # 'results_ilqr/23_02_09_12_50/'
PATH = 'results_offline/' + date_as_path() + '/'
pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)

env = QuadcopterEnv(u0=W0)  # AgentEnv(QuadcopterEnv())
# env.set_time(400, 16)
n_u = len(env.action_space.sample())
n_x = len(env.observation_space.sample())

# env.noise_on = False
dt = env.time[-1] - env.time[-2]
dynamics = ContinuousDynamics(
    f, n_x=n_x, n_u=n_u, u0=W0, dt=dt, method='lsoda')

T = env.steps - 1

cost = OfflineCost(cost=penalty,
                   l_terminal=terminal_penalty,
                   n_x=n_x,
                   n_u=n_u,
                   nu=np.zeros(T),
                   eta=0.01,
                   lamb=np.zeros((T, n_u)),
                   T=T)
# 'results_ilqr/23_01_07_13_56/ilqr_control.npz'
# 'results_ilqr/22_12_31_20_09/ilqr_control.npz'
cost.update_control(file_path=OLD_PATH + 'ilqr_control.npz')
agent = OfflineController(dynamics, cost, T)
expert = LinearAgent(env)


EPISODES = 1

x0 = np.zeros(n_x)
_, us_init, _ = rollout(expert, env, state_init=x0)

agent.x0 = x0
agent.us_init = us_init
xs, us, cost_trace, r = agent.optimize(**PARAMS)
print('ya acabo el ajuste del control')

plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.plot(cost_trace)
# ax.plot(costs[0], color='green', marker='o', linestyle='dashed')
ax.set_title('Costo')
fig.savefig(PATH + 'train_performance.png')
# plt.show()

# Eigen Values
eigvals = np.empty_like(agent._nominal_us)
for i in range(agent.N):
    eigvals[i] = np.linalg.eigvals(agent._C[i])
eig_names = [f'$\lambda_{i}$' for i in range(1, n_u+1)]
fig4, axes = plot_rollouts(eigvals, env.time, eig_names, alpha=0.5)
for j, ax in zip(range(n_u), axes.flatten()):
    ax.hlines(y=0.0, xmin=0, xmax=env.time[-1],
              linewidth=1, color='r', linestyles='dashed')

# Symetry of covariance matrix
vals = [scipy.linalg.issymmetric(agent._C[i]) for i in range(agent.N)]
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
ax.plot(vals)
ax.set_title('Is symetric?')
fig.savefig(PATH + 'is_symetric.png')

create_animation(xs, us, env.time,
                 state_labels=STATE_NAMES,
                 action_labels=ACTION_NAMES,
                 file_name='fitted',
                 path=PATH + 'sample_rollouts/')

agent.reset()
states_, actions_, scores_ = n_rollouts(
    agent, env, n=100)

up = 10 * np.ones(n_x)
down = -10 * np.ones(n_x)
idx = np.apply_along_axis(lambda x: (
    np.less(x, up) & np.greater(x, down)).all(), 1, states_[:, -1])

states = states_[idx]
actions = actions_[idx]
scores = scores_[idx]

fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
fig1.savefig(PATH + 'state_rollouts.png')
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
fig2.savefig(PATH + 'action_rollouts.png')
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
fig3.savefig(PATH + 'score_rollouts.png')

create_report(PATH, 'Ajuste iLQG Offline \n' +
              OLD_PATH, method=None, extra_method='ilqr')
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

print(f'frecuencia de trayectorias estables: {sum(idx) / 100}')
