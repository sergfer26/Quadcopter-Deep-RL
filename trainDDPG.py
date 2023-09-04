import pathlib
import send_email
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from DDPG.ddpg import DDPGagent
from numpy import remainder as rem
from utils import plot_performance
from get_report import create_report
from utils import smooth, date_as_path
from animation import create_animation
from simulation import n_rollouts, plot_rollouts
from dynamics import inv_transform_x, transform_x
from scipy.stats import multivariate_normal
from params import PARAMS_TRAIN_DDPG, STATE_NAMES, ACTION_NAMES, REWARD_NAMES

# from depricated_files.correo import send_correo


BATCH_SIZE = PARAMS_TRAIN_DDPG['BATCH_SIZE']
EPISODES = PARAMS_TRAIN_DDPG['EPISODES']
n = PARAMS_TRAIN_DDPG['n']
TAU = 2 * np.pi  # No es el tau del agente
SHOW = PARAMS_TRAIN_DDPG['SHOW']

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def train(policy: DDPGagent, env: QuadcopterEnv, behavior_policy=None,
          episodes=EPISODES):
    '''
    Argumentos
    ----------
    agent : `DDPG.ddpg.DDPGAgent`
        agente con método DDPG.
    env : `env.QuadcopterEnv`
        entorno de gym.
    episodes : int
        número de episodios a entrenar

    Retornos
    --------
    performance : `dict`
        keys: rewards, policy, critic
    '''
    performance = {k: list() for k in ['rewards', 'policy', 'critic']}
    for episode in range(episodes):
        with tqdm(total=env.steps) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(episodes))
            state = env.reset()
            episode_reward = 0
            while True:
                if behavior_policy is None:
                    action = policy.get_action(state)
                else:
                    action = behavior_policy.get_action(state)
                new_state, reward, done, info = env.step(action)
                episode_reward += reward
                action = info['action']
                policy.memory.push(state, action, reward, new_state, done)
                u, v, w, x, y, z, p, q, r, psi, theta, phi = env.state
                pbar.set_postfix(R='{:.2f}'.format(episode_reward),
                                 psi='{:.2f}'.format(rem(psi, TAU)),
                                 theta='{:.2f}'.format(rem(theta, TAU)),
                                 phi='{:.2f}'.format(rem(phi, TAU)),
                                 z='{:.2f}'.format(z),
                                 y='{:.2f}'.format(y),
                                 x='{:.2f}'.format(x))
                pbar.update(1)
                if len(policy.memory) > BATCH_SIZE:
                    policy_loss, critic_loss = policy.update(BATCH_SIZE)
                    performance['policy'].append(-policy_loss)
                    performance['critic'].append(critic_loss)
                if done:
                    break
                state = new_state
        performance['rewards'].append(episode_reward)
        # avg_rewards.append(np.mean(rewards[-10:]))
    return performance


def main(path):
    plt.style.use("fivethirtyeight")

    env = AgentEnv(QuadcopterEnv(), tx=transform_x, inv_tx=inv_transform_x)
    agent = DDPGagent(env)
    if PARAMS_TRAIN_DDPG['behavior_policy']:
        import torch
        from GPS.policy import Policy
        from params import PARAMS_DDPG
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hidden_sizes = PARAMS_DDPG['hidden_sizes']
        behavior_policy = Policy(env, hidden_sizes)
        behavior_path = PARAMS_TRAIN_DDPG['behavior_path']
        behavior_policy.load(behavior_path)
        agent.actor.load_state_dict(torch.load(
            PARAMS_TRAIN_DDPG['behavior_path'], map_location=device))
    else:
        behavior_policy = None
    performance = train(agent, env, behavior_policy=behavior_policy)
    env.noise_on = False
    agent.save(path)
    smth_rewards = smooth(performance['rewards'], 30)
    fig = plt.figure(figsize=(12, 12), dpi=250)
    gs = fig.add_gridspec(nrows=2, ncols=2)
    ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[0, :])
    plot_performance(smth_rewards, performance['rewards'],
                     xlabel='episodes', ylabel=r'$r_t(\tau)$',
                     title='Entrenamiento',
                     labels=['smooth reward',
                             'episode reward'],
                     ax=ax3
                     )
    for ax, key in zip([ax1, ax2], ['policy', 'critic']):
        ax.plot(performance[key])
        ax.set_title(key + ' loss')

    if SHOW:
        plt.show()
    else:
        fig.savefig(path + 'train_performance.png')  # , bbox_inches='tight')

    states, actions, scores = n_rollouts(agent, env, n, t_x=inv_transform_x)
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
        create_report(path)
    subpath = path + 'sample_rollouts/'
    pathlib.Path(subpath).mkdir(parents=True, exist_ok=True)
    print('Termino de simualcion...')
    create_animation(states, actions, env.time, scores=scores,
                     state_labels=STATE_NAMES,
                     action_labels=ACTION_NAMES,
                     score_labels=REWARD_NAMES,
                     path=subpath
                     )
    agent.save(path)
    # keys: rewards, policy, critic
    np.savez(path + 'performance.npz', performance)
    return path


if __name__ == "__main__":
    PATH = 'results_ddpg/' + date_as_path() + '/'
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
    if not SHOW:
        send_email.report_sender(main, args=[PATH])
    else:
        main(PATH)
