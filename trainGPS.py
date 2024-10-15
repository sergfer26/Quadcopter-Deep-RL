import time
import pathlib
import send_email
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
from Linear.equations import f, W0
from utils import date_as_path
from GPS.utils import PolicyNoise
from utils import plot_performance, violin_plot
from GPS import GPS, Policy, ContinuousDynamics, iLQG
from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from dynamics import transform_x, transform_u
from dynamics import inv_transform_u, inv_transform_x
from dynamics import penalty, terminal_penalty
from params import PARAMS_TRAIN_GPS as PARAMS
from params import PARAMS_DDPG
from GPS.params import PARAMS_OFFLINE
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES, state_cmap
from params import PARAMS_OBS
from simulation import plot_rollouts, n_rollouts, rollout
from animation import create_animation
from get_report import create_report
# from mycolorpy import colorlist as mcp


SHOW = PARAMS['SHOW']

# if not SHOW:
#     from functools import partialmethod
#     tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@dataclass
class GPSResult:
    loss: np.ndarray
    nu: np.ndarray
    eta: np.ndarray
    lamb: np.ndarray
    policy_div: np.ndarray
    control_div: np.ndarray
    policy_cost: np.ndarray
    control_cost: np.ndarray
    policy_states: np.ndarray
    control_states: np.ndarray


def train_gps(gps: GPS, K, path, per_kl=0.1,
              constrained_actions=False,
              shuffle_batches=True, policy_updates=2):  # , x0=None):
    loss = np.empty(K)
    nu = np.empty((K, gps.N, gps.T))
    eta = np.empty(K)
    lamb = np.empty((K, gps.N, gps.T, gps.n_u))
    policy_div = np.empty((K, gps.N, gps.M))
    control_div = np.empty((K, gps.N))
    # Inicializa x0s
    # if isinstance(x0, np.ndarray):
    #     gps.x0 = x0
    # else:
    #     gps.init_x0()

    policy_cost = np.empty((K, gps.M))
    policy_states = np.empty((K, gps.M, gps.n_x))
    control_cost = np.empty((K, gps.M))
    control_states = np.empty((K, gps.M, gps.n_x))
    dynamics = ContinuousDynamics(**gps.dynamics_kwargs)
    control = iLQG(dynamics, None, gps.T)
    with tqdm(total=K) as pbar:
        for k in range(K):
            pbar.set_description(f'Update {k + 1}/'+str(K))
            loss[k], policy_div[k], control_div[k] = gps.update_policy(
                path, constrained_actions, shuffle_batches, policy_updates)
            nu[k] = gps.nu  # np.mean(gps.nu, axis=(0, 1))
            eta[k] = np.mean(gps.eta, axis=0)
            lamb[k] = gps.lamb  # np.mean(gps.lamb, axis=(0, 1))
            lamb_ = np.linalg.norm(np.mean(gps.lamb, axis=(0, 1)))
            pbar.set_postfix(loss='{:.2f}'.format(loss[k]),
                             kl_step='{:.2f}'.format(gps.kl_step),
                             lamb=lamb_,
                             eta='{:.2f}'.format(eta[k]),
                             nu='{:.3f}'.format(np.mean(gps.nu, axis=(0, 1))),
                             policy_div=0.5 * np.mean(policy_div[k]))
            pbar.update(1)
            gps.kl_step = gps.kl_step * (1 - per_kl)
            # Validación de modelos
            x0, indices = select_x0(gps, gps.M, return_indices=True)
            control_cost[k], control_states[k] = lqr_rollouts(
                x0, indices, control, gps.env, path + 'buffer/')

            x0 = np.apply_along_axis(gps.t_x, -1, x0)
            states, _, scores = n_rollouts(gps.policy, gps.policy.env,
                                           gps.M, t_x=gps.inv_t_x,
                                           states_init=x0)
            policy_states[k] = states[:, -1]
            policy_cost[k] = scores[:, -1, 1]
    return GPSResult(loss, nu, eta, lamb, policy_div, control_div,
                     policy_cost, control_cost, policy_states, control_states)


def lqr_rollouts(x0: np.ndarray, indices: np.ndarray, control: iLQG, env, path):
    n = len(indices)
    episode_rewards = np.empty(n)
    last_states = np.empty((n, x0.shape[-1]))
    for e, i in enumerate(indices):
        control.load(path, file_name=f'control_{i}.npz')
        states, _, scores = rollout(control, env, state_init=x0[e])
        episode_rewards[e] = scores[-1, 1]
        last_states = states[-1]
    return episode_rewards, last_states


def select_x0(gps: GPS, n, return_indices=False):
    if gps.N > 1:
        indices = np.random.choice(gps.N, n)
        state_init = gps.x0[indices]
        x0 = np.apply_along_axis(gps._random_x0, -1, state_init, 1)
        x0 = np.squeeze(x0, axis=1)
    else:
        x0 = gps._random_x0(gps.x0, n)
    if return_indices:
        return x0, indices
    return x0


def main(path):
    # 1. Creación de instancias
    K = PARAMS['UPDATES']
    rollouts = PARAMS['M']
    samples = PARAMS['samples']
    KL_STEP = PARAMS_OFFLINE['kl_step']
    # 1.1 Dynamics
    env = QuadcopterEnv()
    dt = env.dt
    n_u = env.action_space.shape[0]
    n_x = env.observation_space.shape[0]
    other_env = AgentEnv(env, tx=transform_x, inv_tx=inv_transform_x)
    other_env.noise_on = False
    hidden_sizes = PARAMS_DDPG['hidden_sizes']
    policy = Policy(other_env, hidden_sizes)
    other_env.noise = PolicyNoise(policy)
    dynamics_kwargs = dict(f=f, n_x=n_x, n_u=n_u, dt=dt, u0=W0)
    # 1.2 GPS
    high_range = np.array(
        [.0, .0, .0, 1., 1., 1., .0, .0, .0, np.pi/64, np.pi/64, np.pi/64])
    low_range = - high_range
    states = np.load(
        "results_ilqr/stability_analysis/23_07_14_11_30/stability_region.npz")['states']
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
              is_stochastic=PARAMS['is_stochastic'],
              time_step=PARAMS['time_step'],
              states=states)
    ti = time.time()
    # 2. Training
    result = train_gps(gps, K, path, per_kl=PARAMS_OFFLINE['per_kl'],
                       shuffle_batches=PARAMS['shuffle_batches'],
                       policy_updates=PARAMS['policy_updates']
                       )
    # x0=np.load('states_init.npz')['states_init'])
    np.savez(path + 'results.npz',
             policy_div=result.policy_div,
             control_div=result.policy_div,
             policy_cost=result.policy_cost,
             control_cost=result.control_cost,
             policy_states=result.policy_states,
             control_states=result.control_states
             )

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
    plot_performance(result.loss, xlabel='iteraciones',
                     ylabel='$L_{\\theta}(\\theta, p)$',
                     title='Entrenamiento', ax=ax1)
    # 3.2 eta's and nu's evolution
    dic = {'$\eta$': result.eta, '$\\nu$': np.mean(result.nu, axis=(1, 2))}
    for ax, key in zip([ax2, ax3], dic.keys()):
        ax.plot(dic[key])
        ax.set_title(key)
    # 3.3 lambdas
    LAMB_NAMES = [f'$\lambda_{i}$' for i in range(1, n_u+1)]
    ax = np.array([ax41, ax42, ax43, ax44])
    plot_rollouts(np.mean(result.lamb, axis=(1, 2)),
                  env.time, LAMB_NAMES, axes=ax)
    fig.savefig(path + 'train_performance.png')

    # 3.4 Policy's simulations
    states_init = np.apply_along_axis(
        gps.t_x, -1, select_x0(gps, rollouts))
    states, actions, scores = n_rollouts(
        policy, other_env, rollouts, t_x=inv_transform_x,
        states_init=states_init)
    print('Terminó de simualación...')

    up = 10 * np.ones(n_x)
    down = -10 * np.ones(n_x)
    idx = np.apply_along_axis(lambda x: (
        np.less(x, up) & np.greater(x, down)).all(), 1, states[:, -1])

    if sum(idx) > samples:
        states = states[idx]
        actions = actions[idx]
        scores = scores[idx]

    try:
        print(f'abs(state) < 10 -> {sum(idx) / len(idx)}')
        up = np.ones(n_x)
        down = -np.ones(n_x)
        idx = np.apply_along_axis(lambda x: (
            np.less(x, up) & np.greater(x, down)).all(), 1, states[:, -1])
        print(f'abs(state) < 1 -> {sum(idx)/len(idx)}')
    except:
        print('error en cuentas de estados')

    state_names = ['$x$', '$u$',
                   '$y$', '$v$',
                   '$z$', '$w$',
                   '$\\varphi$', '$p$',
                   r'$\theta$',  '$q$',
                   '$\psi$', '$r$']
    eps = 0.25
    indices = [state_names.index(label) for label in STATE_NAMES]
    params_obs = {k: eval(PARAMS_OBS[k]) for k in state_names}
    state_ylims = np.array([
        [val + eps, - val - eps] if abs(val) > 1 else [1, -1]
        for val in params_obs.values()
    ])
    fig1, _ = plot_rollouts(
        states[:, :, indices], env.time, state_names, alpha=0.05,
        ylims=state_ylims)
    fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
    fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
    fig1.savefig(path + 'state_rollouts.png')
    fig2.savefig(path + 'action_rollouts.png')
    fig3.savefig(path + 'score_rollouts.png')

    fig4, _ = plot_rollouts(result.lamb[-1], env.time, LAMB_NAMES, alpha=0.2)
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
    states_control = states_control.reshape((N * M, T + 1, n_x))
    fig1, _ = plot_rollouts(states_control[:, :, indices],
                            env.time, state_names, alpha=0.005,
                            ylims=state_ylims)
    fig2, _ = plot_rollouts(actions_control.reshape((N * M, T, n_u)),
                            env.time, ACTION_NAMES, alpha=0.005)
    fig1.savefig(path + 'buffer/state_rollouts.png')
    fig2.savefig(path + 'buffer/action_rollouts.png')

    try:
        # 3.4 Mean and std div evolution
        fig, axes = plt.subplots(figsize=(10, 5), ncols=2, dpi=250)
        violin_plot(policy_div=np.mean(result.policy_div, axis=-1).T,
                    x_name='iteraciones',
                    y_name='$Div(\pi, p)$', ax=axes[0],
                    )
        violin_plot(control_div=result.control_div.T,
                    x_name='iteraciones',
                    y_name='$Div(p, \hat p)$', ax=axes[1]
                    )
        fig.savefig(path + 'div_updates.png')
    except:
        print('fallo div_updates.png')

    try:
        # 3.5 Mean and std cost evolution
        fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
        violin_plot(policy=result.policy_cost.T,
                    control=result.control_cost.T,
                    x_name='iteraciones',
                    y_name='$c(\\tau)$',
                    hue='cost', ax=ax
                    )
        fig.savefig(path + 'cost_updates.png')
    except:
        print('fallo cost_updates.png')

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
                     goal=np.zeros(3),
                     title='Guided Policy Search: policy $\pi_{\\theta}$',
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
