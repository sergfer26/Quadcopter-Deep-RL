import traceback
import numpy as np
import pathlib
import send_email
from GPS.utils import ContinuousDynamics
from Linear.equations import f, W0, omega_0
from env import QuadcopterEnv
from GPS.controller import OfflineController, iLQG
from simulation import plot_rollouts, n_rollouts
from matplotlib import pyplot as plt
from animation import create_animation
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES
from get_report import create_report
from utils import date_as_path
from dynamics import penalty, terminal_penalty, transform_x, inv_transform_u
from GPS.utils import OfflineCost
from GPS.params import PARAMS_OFFLINE as PARAMS
from utils import plot_performance
from ilqr.cost import FiniteDiffCost


def main(updates, path, old_path,
         adaptive_kl=False,
         per_kl=0.1,
         kl_init=200,
         max_eta=1e8
         ):

    env = QuadcopterEnv(u0=W0)
    n_u = len(env.action_space.sample())
    n_x = len(env.observation_space.sample())

    dt = env.time[-1] - env.time[-2]
    dynamics = ContinuousDynamics(
        f, n_x=n_x, n_u=n_u, u0=W0, dt=dt)

    T = env.steps - 1

    cost = OfflineCost(cost=penalty,
                       l_terminal=terminal_penalty,
                       n_x=n_x,
                       n_u=n_u,
                       nu=PARAMS['nu'] * np.ones(T),
                       eta=eval(PARAMS['min_eta']),
                       lamb=PARAMS['lamb'] * np.ones((T, n_u)),
                       T=T)
    # 'results_ilqr/23_01_07_13_56/ilqr_control.npz'
    # 'results_ilqr/22_12_31_20_09/ilqr_control.npz'
    other_cost = FiniteDiffCost(l=penalty,
                                l_terminal=terminal_penalty,
                                state_size=n_x,
                                action_size=n_u
                                )
    cost.update_policy(file_path='models/policy.pt',
                       t_x=transform_x, inv_t_u=inv_transform_u,
                       cov=omega_0 * np.identity(n_u))
    agent = OfflineController(dynamics, cost, T)
    expert = iLQG(dynamics, other_cost, T, is_stochastic=False)
    expert.load(old_path)
    agent.alpha = expert.alpha
    agent.check_constrain = True

    agent.x0 = env.observation_space.sample()
    print(f'x0={agent.x0}')  # np.zeros(n_x)

    agent.us_init = expert.rollout(agent.x0)[1]
    cost.update_control(control=expert)
    print(f'valor inicial de eta={cost.eta}')
    etas = [cost.eta]
    div = []
    failed = False
    kl_step = kl_init
    for i in range(updates):
        try:
            min_eta = cost.eta
            xs, us, cost_trace, kl_div = agent.optimize(
                kl_step, min_eta=min_eta, max_eta=max_eta)
            agent.us_init = agent.rollout(agent.x0)[1]
            kl_step = kl_div if adaptive_kl else kl_step * (1 - per_kl)
            agent.check_constrain = False
            etas.append(min_eta)
            div.append(kl_div)
            cost.update_control(agent)
            agent.save(path, file_name=f'control_{i}.npz')
        except Exception:
            failed = True
            print(traceback.format_exc())
            agent.save(path, file_name=f'failed_{i}.npz')
            print(f'fallo en la iteración {i}')
            break

    if not failed or i > 0:
        print(
            f'ya acabo el ajuste del control, eta={min_eta}, kl_div={kl_div}')

        plt.style.use("fivethirtyeight")
        fig = plt.figure(figsize=(12, 12), dpi=250)
        gs = fig.add_gridspec(nrows=12, ncols=4)
        ax1, ax2 = fig.add_subplot(gs[0:4, :2]), fig.add_subplot(gs[4:8, :2])
        ax4 = fig.add_subplot(gs[8:, :2])
        ax31 = fig.add_subplot(gs[0:3, 2:])
        ax32 = fig.add_subplot(gs[3:6, 2:])
        ax33 = fig.add_subplot(gs[6:9, 2:])
        ax34 = fig.add_subplot(gs[9:, 2:])
        ax3 = np.array([ax31, ax32, ax33, ax34])

        # 3.1 Loss' plot
        plot_performance(etas, xlabel='iteraciones',
                         ylabel='$\eta$', ax=ax2, labels=['$\eta$'])
        plot_performance(div, xlabel='iteraciones',
                         ylabel='divergencia', ax=ax4, labels=['$KL(p||\hat p)$'])
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
    adaptive_kl = PARAMS['adaptive_kl']
    per_kl = PARAMS['per_kl']
    kl_step = PARAMS['kl_step']
    max_eta = eval(PARAMS['max_eta'])
    send_email.report_sender(main, args=[updates, PATH, OLD_PATH, adaptive_kl,
                                         per_kl, kl_step, max_eta])
    # main(updates, PATH, OLD_PATH)
