import numpy as np
import pathlib
import send_email
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
from utils import plot_performance


def main(updates, path, old_path):

    env = QuadcopterEnv(u0=W0)
    n_u = len(env.action_space.sample())
    n_x = len(env.observation_space.sample())

    dt = env.time[-1] - env.time[-2]
    dynamics = ContinuousDynamics(
        f, n_x=n_x, n_u=n_u, u0=W0, dt=dt, method='lsoda')

    T = env.steps - 1

    cost = OfflineCost(cost=penalty,
                       l_terminal=terminal_penalty,
                       n_x=n_x,
                       n_u=n_u,
                       nu=np.zeros(T),
                       eta=1e-4,
                       lamb=np.zeros((T, n_u)),
                       T=T)
    # 'results_ilqr/23_01_07_13_56/ilqr_control.npz'
    # 'results_ilqr/22_12_31_20_09/ilqr_control.npz'
    cost.update_control(file_path=old_path + 'ilqr_control.npz')
    agent = OfflineController(dynamics, cost, T)
    expert = LinearAgent(env)

    x0 = np.zeros(n_x)
    _, us_init, _ = rollout(expert, env, state_init=x0)

    agent.x0 = x0
    agent.us_init = us_init
    min_eta = cost.eta
    print(f'valor inicial de eta={min_eta}')
    etas = [min_eta]
    kl_div = PARAMS['kl_step']
    for _ in range(updates):
        xs, us, cost_trace, r, kl_div = agent.optimize(
            kl_div, min_eta=min_eta)
        min_eta = r.root
        etas.append(min_eta)
        _, us_init = agent.rollout(x0)
        agent.us_init = us_init
        cost.update_control(agent)

    print(f'ya acabo el ajuste del control, eta={min_eta}, kl_div={kl_div}')

    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=(14, 12), dpi=250)
    gs = fig.add_gridspec(nrows=4, ncols=4)
    ax1, ax2 = fig.add_subplot(gs[0:2, :2]), fig.add_subplot(gs[2:, :2])
    ax3 = fig.add_subplot(gs[:, 2:])

    # 3.1 Loss' plot
    plot_performance(etas, xlabel='iteraciones',
                     ylabel='$\eta$', ax=ax2)
    # Análisis de los eigen valores de la matriz de control
    eigvals = np.linalg.eigvals(agent._C)
    eig_names = [f'$\lambda_{i}$' for i in range(1, n_u+1)]
    plot_rollouts(eigvals, env.time, eig_names, alpha=0.5, ax=ax3)

    ax1.plot(cost_trace)
    ax1.set_title('Costo')
    fig.savefig(path + 'train_performance.png')

    mask = np.apply_along_axis(np.greater, -1, eigvals, np.zeros(n_u))
    if mask.all():
        print('Todas las matrices C son positivas definidas')
    else:
        indices = np.apply_along_axis(lambda x: x.any(), -1, ~mask)
        print(f'{indices.sum()} no son positivas definidas')
        np.savez(
            PATH + 'invalid_C.npz',
            C=agent._C[indices]
        )

    create_animation(xs, us, env.time,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     file_name='fitted',
                     path=path + 'sample_rollouts/')

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
    fig1.savefig(path + 'state_rollouts.png')
    fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
    fig2.savefig(path + 'action_rollouts.png')
    fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
    fig3.savefig(path + 'score_rollouts.png')

    create_report(path, 'Ajuste iLQG Offline \n' +
                  old_path, method=None, extra_method='ilqr')
    agent.save(path)
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
                     path=path + 'sample_rollouts/')
    return PATH


if __name__ == '__main__':
    # 'results_ilqr/23_02_08_22_01/'  # 'results_ilqr/23_02_09_12_50/'
    # OLD_PATH = 'results_offline/23_02_16_13_50/'
    OLD_PATH = 'results_ilqr/23_02_16_23_21/'
    PATH = 'results_offline/' + date_as_path() + '/'
    pathlib.Path(PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)
    updates = 5
    send_email.report_sender(main, args=[updates, PATH, OLD_PATH])
    # main(updates, PATH, OLD_PATH)
