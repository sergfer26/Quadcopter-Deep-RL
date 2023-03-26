import time
import pathlib
import send_email
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Linear.equations import f, W0
from utils import date_as_path
from utils import plot_performance
from GPS import GPS, Policy
from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from dynamics import transform_x, transform_u
from dynamics import inv_transform_u, inv_transform_x
from dynamics import penalty, terminal_penalty
from params import PARAMS_TRAIN_GPS as PARAMS
from params import PARAMS_DDPG
from GPS.params import PARAMS_OFFLINE
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from simulation import plot_rollouts, n_rollouts
from animation import create_animation
from get_report import create_report


SHOW = PARAMS['SHOW']

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def train_gps(gps: GPS, K, path, per_kl=0.1, constrained_actions=False):
    losses = np.empty(K)
    nus = np.empty((K, gps.N, gps.T))
    etas = np.empty(K)
    lambdas = np.empty((K, gps.N, gps.T, gps.n_u))
    # Inicializa x0s
    gps.init_x0()
    with tqdm(total=K) as pbar:
        for k in range(K):
            pbar.set_description(f'Update {k + 1}/'+str(K))
            loss, div = gps.update_policy(path, constrained_actions)
            mean_div = 0.5 * np.mean(div)
            losses[k] = loss
            nus[k] = gps.nu  # np.mean(gps.nu, axis=(0, 1))
            etas[k] = np.mean(gps.eta, axis=0)
            lambdas[k] = gps.lamb  # np.mean(gps.lamb, axis=(0, 1))
            lamb = np.linalg.norm(np.mean(gps.lamb, axis=(0, 1)))
            pbar.set_postfix(loss='{:.2f}'.format(loss),
                             kl_step='{:.2f}'.format(gps.kl_step),
                             lamb=lamb,
                             eta='{:.2f}'.format(etas[k]),
                             nu='{:.3f}'.format(np.mean(gps.nu, axis=(0, 1))),
                             mean_div=mean_div)
            pbar.update(1)
            gps.kl_step = gps.kl_step * (1 - per_kl)
    return losses, nus, etas, lambdas, div


def select_x0(gps, n):
    if gps.N > 1:
        indices = np.random.choice(gps.N, n)
        state_init = gps.x0[indices]
        x0 = np.apply_along_axis(gps._random_x0, -1, state_init, 1)
        x0 = np.squeeze(x0, axis=1)
    else:
        x0 = gps._random_x0(gps.x0, n)
    return x0


def main(path):
    # 1. Creación de instancias
    K = PARAMS['UPDATES']
    rollouts = PARAMS['M']
    samples = PARAMS['samples']
    KL_STEP = PARAMS_OFFLINE['kl_step']
    # 1.1 Dynamics
    env = QuadcopterEnv()
    dt = env.time[-1] - env.time[-2]
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    hidden_sizes = PARAMS_DDPG['hidden_sizes']
    policy = Policy(other_env, hidden_sizes)
    dynamics_kwargs = dict(f=f, n_x=n_x, n_u=n_u,
                           dt=dt, u0=W0)
    # 1.2 GPS
    high_range = np.array(
        [.0, .0, .0, 1.5, 1.5, 1.5, .0, .0, .0, np.pi/32, np.pi/32, np.pi/32])
    low_range = - high_range
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
              eta=eval(PARAMS_OFFLINE['min_eta']),
              nu=eval(PARAMS_OFFLINE['nu']),
              lamb=eval(PARAMS_OFFLINE['lamb']),
              alpha_lamb=eval(PARAMS_OFFLINE['alpha_lamb']),
              learning_rate=eval(PARAMS_DDPG['actor_learning_rate']),
              kl_step=KL_STEP,
              init_sigma=W0[0],
              low_range=low_range,
              high_range=high_range,
              batch_size=PARAMS['batch_size'],
              is_stochastic=PARAMS['is_stochastic']
              )
    ti = time.time()
    # 2. Training
    losses, nus, etas, lambdas, div = train_gps(
        gps, K, PATH, per_kl=PARAMS_OFFLINE['per_kl'])
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
    dic = {'$\eta$': etas, '$\\nu$': np.mean(nus, axis=(1, 2))}
    for ax, key in zip([ax2, ax3], dic.keys()):
        ax.plot(dic[key])
        ax.set_title(key)
    # 3.3 lambdas
    LAMB_NAMES = [f'$\lambda_{i}$' for i in range(1, n_u+1)]
    ax = np.array([ax41, ax42, ax43, ax44])
    plot_rollouts(np.mean(lambdas, axis=(1, 2)), env.time, LAMB_NAMES, ax=ax)
    fig.savefig(path + 'train_performance.png')

    # 3.4 Policy's simulations
    states_init = np.apply_along_axis(
        gps.t_x, -1, select_x0(gps, rollouts))
    states, actions, scores = n_rollouts(
        policy, other_env, rollouts, t_x=inv_transform_x,
        states_init=states_init)
    print('Terminó de simualación...')

    fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
    fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
    fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
    fig1.savefig(path + 'state_rollouts.png')
    fig2.savefig(path + 'action_rollouts.png')
    fig3.savefig(path + 'score_rollouts.png')

    fig4, _ = plot_rollouts(lambdas[-1], env.time, LAMB_NAMES, alpha=0.2)
    fig4.savefig(path + 'lambdas.png')

    N = gps.N
    M = gps.M
    T = gps.T

    # 3.5 Control's simulations
    states_control = np.empty((N, M, T + 1, n_x))
    actions_control = np.empty((N, M, T, n_u))
    for i in range(gps.N):
        file = np.load(path + f'buffer/rollouts_{i}.npz')
        states_control[i] = file['xs']
        actions_control[i] = file['us']
    fig1, _ = plot_rollouts(states_control.reshape((N * M, T + 1, n_x)),
                            env.time, STATE_NAMES, alpha=0.005)
    fig2, _ = plot_rollouts(actions_control.reshape((N * M, T, n_u)),
                            env.time, ACTION_NAMES, alpha=0.005)
    fig1.savefig(path + 'buffer/state_rollouts.png')
    fig2.savefig(path + 'buffer/action_rollouts.png')

    # 3.4 Divergence's along time
    mean_div = np.mean(div, axis=(0, 1))  # (T,)
    std_div = np.std(div, axis=(0, 1))  # (T, )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(env.time[:T], mean_div, alpha=0.6, color='blue',
            label='current', linewidth=2.0)
    ax.fill_between(env.time[:T], mean_div + std_div, mean_div - std_div,
                    color='lightskyblue', alpha=0.4)
    ax.fill_between(env.time[:T], mean_div + 2 * std_div, mean_div + 2 * std_div,
                    color='lightskyblue', alpha=0.2)
    ax.legend(loc='best')
    ax.set_ylabel("divergence")
    ax.set_xlabel("$t$ (s)")
    fig.savefig(path + 'kl_div.png')

    # 4. Creación de reporte
    create_report(path, title='Entrenamiento GPS',
                  method='gps', extra_method='ilqr')

    subpath = path + 'sample_rollouts/'
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)

    sample_indices = np.random.randint(states.shape[0], size=samples)
    states_samples = states[sample_indices]
    actions_samples = actions[sample_indices]
    scores_samples = scores[sample_indices]

    # 5. Creación de animación
    print('Creando animación...')
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
