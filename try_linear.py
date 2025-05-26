import numpy as np

from matplotlib import pyplot as plt

from env import QuadcopterEnv
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES

from Linear.agent import LinearAgent
from simulation import n_rollouts, plot_rollouts


env = QuadcopterEnv()
agent = LinearAgent(env)
array = np.load(
    "results_linear/stability_analysis/24_07_28_13_56/stability_10000.npz")
# states_init = array['states'][0, :, 0]
states_init = None
states, actions, scores = n_rollouts(
    agent, env, 100, states_init=states_init)
fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
# fig1.savefig('state_rollouts.png')
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
# fig2.savefig('action_rollouts.png')
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
# fig3.savefig('score_rollouts.png')
plt.show()
