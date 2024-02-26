import numpy as np
import pathlib
import matplotlib.pyplot as plt
from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from animation import create_animation
from simulation import n_rollouts, plot_rollouts, rollout
from dynamics import inv_transform_x, transform_x
from params import PARAMS_TRAIN_DDPG, STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from GPS.policy import Policy
from params import PARAMS_DDPG
from GPS.controller import DummyController
from utils import date_as_path


state_names = ['$x$', '$u$',
               '$y$', '$v$',
               '$z$', '$w$',
               '$\\varphi$', '$p$',
               r'$\theta$',  '$q$',
               '$\psi$', '$r$']
indices = [state_names.index(label) for label in STATE_NAMES]


n = 100
PATH = 'test_noise/' + date_as_path() + '/'
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
control = DummyController(
    'results_gps/23_04_22_02_26/buffer/', 'control_0.npz')
env = AgentEnv(QuadcopterEnv(), tx=transform_x, inv_tx=inv_transform_x)
hidden_sizes = PARAMS_DDPG['hidden_sizes']
policy = Policy(env, hidden_sizes)
path = PARAMS_TRAIN_DDPG['behavior_path']
policy.load(path)
ylims = np.array([
    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-9, 9],
    [-9, 9],
    [-9, 9],
    [-1, 1],
    [-1, 1],
    [-1, 1],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2]
])
ylims = ylims[indices]
# env = QuadcopterEnv()
# curve = np.apply_along_axis(
#     lambda t: 0.01 * np.array([t * np.cos(t), t * np.sin(t), t]), -1, env.time
# )
#
#
# aux = np.zeros((env.steps + 1, env.observation_space.shape[0]))
# aux[:, 3:6] = curve.T
# states, actions, scores = rollout(
#     control, env, curve=aux)

# states = np.apply_along_axis(inv_transform_x, -1, states)
# fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
# plt.show()

# create_animation(states, actions, env.time,
#                  scores=scores,
#                  state_labels=STATE_NAMES,
#                  action_labels=ACTION_NAMES,
#                  score_labels=REWARD_NAMES,
#                  goal=np.zeros(3),
#                  title='iLQR path following',
#                  path='curves/',
#                  file_name='path_follow'
#                  )

# 1. Noise OU
states, actions, scores = n_rollouts(policy, env, n, t_x=inv_transform_x)
fig1, _ = plot_rollouts(states[:, :, indices], env.time, state_names,
                        alpha=0.05, ylims=ylims)
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
fig1.savefig(PATH + 'state_noisy.png')
fig2.savefig(PATH + 'action_noisy.png')
fig3.savefig(PATH + 'score_noisy.png')

# 2. Noiseless
init_states = states[:, 0]
env.noise_on = False
states, actions, scores = n_rollouts(
    policy, env, n, t_x=inv_transform_x,
    states_init=np.apply_along_axis(transform_x, -1, init_states)
)
fig1, _ = plot_rollouts(states[:, :, indices], env.time, state_names,
                        alpha=0.1, ylims=ylims)
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.1)
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.1)
fig1.savefig(PATH + 'state_noiseless.png')
fig2.savefig(PATH + 'action_noiseless.png')
fig3.savefig(PATH + 'score_noiseless.png')

# 3. With policy's sigma

policy.is_stochastic = True
C = np.empty(
    (7, env.steps, env.action_space.shape[0], env.action_space.shape[0])
)
nu = np.empty(
    (7, env.steps)
)
for i in range(7):
    arrays = np.load(path + f'buffer/control_{i}.npz')
    C[i] = arrays['C']
    nu[i] = arrays['nu']

a = np.sum(nu, axis=(0, 1)) ** (-1)
b = np.einsum('ij, ijkl->kl', nu, np.linalg.inv(C))
policy._C = a * np.linalg.inv(b) + 1e-2 * np.identity(4)
policy.is_stochastic = True

states, actions, scores = n_rollouts(
    policy, env, n, t_x=inv_transform_x,
    states_init=np.apply_along_axis(transform_x, -1, init_states)
)
fig1, _ = plot_rollouts(states[:, :, indices], env.time, state_names,
                        alpha=0.1, ylims=ylims)
fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.1)
fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.1)
fig1.savefig(PATH + 'state_stochastic.png')
fig2.savefig(PATH + 'action_stochastic.png')
fig3.savefig(PATH + 'score_stochastic.png')
