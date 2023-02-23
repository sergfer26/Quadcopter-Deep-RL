import pathlib
import time
import numpy as np
import send_email
from tqdm import tqdm
from utils import date_as_path
from utils import plot_performance
import matplotlib.pyplot as plt
from Linear.equations import f, W0
from GPS import GPS, Policy
from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from dynamics import transform_x, transform_u
from dynamics import inv_transform_u, inv_transform_x
from dynamics import penalty, terminal_penalty
from params import PARAMS_TRAIN_GPS as PARAMS
from GPS.params import PARAMS_OFFLINE
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from simulation import plot_rollouts, n_rollouts
from animation import create_animation
from get_report import create_report


SHOW = PARAMS['SHOW']
KL_STEP = PARAMS_OFFLINE['kl_step']

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def train_gps(gps: GPS, K, path):
    losses = np.empty(K)
    nus = np.empty((K, gps.T))
    etas = np.empty((K, gps.T))
    lambdas = np.empty((K, gps.T, gps.n_u))
    with tqdm(total=K) as pbar:
        for k in range(K):
            pbar.set_description(f'Update {k + 1}/'+str(K))
            loss, _ = gps.update_policy(path)
            losses[k] = loss
            nus[k] = np.mean(gps.nu, axis=0)
            etas[k] = np.mean(gps.eta, axis=0)
            lambdas[k] = np.mean(gps.lamb, axis=0)
            pbar.set_postfix(loss='{:.2f}'.format(loss))
            pbar.update(1)
    return losses, nus, etas, lambdas


def main(path):
    K = PARAMS['UPDATES']
    rollouts = PARAMS['rollouts']
    samples = PARAMS['samples']
    # 1. Setup
    env = QuadcopterEnv()
    dt = env.time[-1] - env.time[-2]
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    policy = Policy(other_env, [64, 64])
    # policy.load_state_dict(torch.load(
    #     'results_ddpg/12_9_113/actor', map_location='cpu'))
    dynamics_kwargs = dict(f=f, n_x=n_x, n_u=n_u,
                           dt=dt, u0=W0)
    # 2. Training
    ti = time.time()

    gps = GPS(env,
              policy,
              dynamics_kwargs,
              penalty,
              terminal_penalty,
              t_x=transform_x,
              t_u=transform_u,
              inv_t_x=inv_transform_x,
              inv_t_u=inv_transform_u,
              N=PARAMS['N'],
              M=PARAMS['M'],
              eta=PARAMS['eta'],
              nu=PARAMS['nu'],
              lamb=PARAMS['lamb'],
              kl_step=KL_STEP
              )
    losses, nus, etas, lambdas = train_gps(gps, K, PATH)
    tf = time.time()
    print(f'tiempo de ajuste de política por GPS: {tf - ti}')
    policy.save(path)
    # 3. Graphs
    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=(16, 12), dpi=250)
    gs = fig.add_gridspec(nrows=4, ncols=3)
    ax1 = fig.add_subplot(gs[:3, :2])
    ax2, ax3 = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    ax41 = fig.add_subplot(gs[0, -1])
    ax42 = fig.add_subplot(gs[1, -1])
    ax43 = fig.add_subplot(gs[2, -1])
    ax44 = fig.add_subplot(gs[3, -1])
    # 3.1 Loss' plot
    plot_performance(losses, xlabel='iteraciones',
                     ylabel='$L_{\\theta}(\\theta, p)$',
                     title='Entrenamiento', ax=ax1)
    # 3.2 eta's and nu's evolution
    dic = {'$eta$': etas, '$nu$': nus}
    for ax, key in zip([ax2, ax3], dic.keys()):
        ax.plot(dic[key])
        ax.set_title(key)
    # 3.3 lambdas
    index = np.linspace(1, K, K)
    labels = [f'$\lambda_{i}$' for i in range(1, n_u+1)]
    ax = np.array([ax41, ax42, ax43, ax44])
    plot_rollouts(lambdas, index, labels, ax=ax)
    if SHOW:
        plt.show()
    else:
        fig.savefig(path + 'train_performance.png')
    # 4. Simulation
    states, actions, scores = n_rollouts(
        policy, other_env, rollouts, t_x=inv_transform_x)
    fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
    fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
    fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
    if SHOW:
        plt.show()
    else:
        fig1.savefig(path + 'state_rollouts.png')
        fig2.savefig(path + 'action_rollouts.png')
        fig3.savefig(path + 'score_rollouts.png')
    if not SHOW:
        create_report(path, title='Entrenamiento MPC-GPS',
                      method='gps', extra_method='ilqr')
    subpath = path + 'sample_rollouts/'
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
    print('Terminó de simualación...')

    sample_indices = np.random.randint(states.shape[0], size=samples)
    states_samples = states[sample_indices]
    actions_samples = actions[sample_indices]
    scores_samples = scores[sample_indices]

    create_animation(states_samples, actions_samples, env.time,
                     scores=scores_samples,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     score_labels=REWARD_NAMES,
                     path=subpath
                     )
    return path


if __name__ == '__main__':
    PATH = 'results_gps/' + date_as_path() + '/'
    pathlib.Path(PATH + 'buffer/').mkdir(parents=True, exist_ok=True)
    if not SHOW:
        send_email.report_sender(main, args=[PATH])
    else:
        main(PATH)
