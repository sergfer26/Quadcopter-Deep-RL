import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from numpy import remainder as rem
from env import QuadcopterEnv, AgentEnv
from DDPG.ddpg import DDPGagent
from get_report import create_report_ddpg
from params import PARAMS_TRAIN_DDPG, STATE_NAMES, ACTION_NAMES, SCORE_NAMES
from simulation import nSim, plot_nSim2D, plot_nSim3D
from utils import smooth, date_as_path
from correo import send_correo
from utils import plot_performance


BATCH_SIZE = PARAMS_TRAIN_DDPG['BATCH_SIZE']
EPISODES = PARAMS_TRAIN_DDPG['EPISODES']
n = PARAMS_TRAIN_DDPG['n']
TAU = 2 * np.pi  # No es el tau del agente
SHOW = PARAMS_TRAIN_DDPG['SHOW']

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def train(agent, env, episodes=EPISODES):
    rewards = []
    avg_rewards = []
    for episode in range(episodes):
        with tqdm(total=env.steps) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(episodes))
            state = env.reset()
            episode_reward = 0
            while True:
                action = agent.get_action(state)
                new_state, reward, done, info = env.step(action)
                episode_reward += reward
                action = info['action']
                agent.memory.push(state, action, reward, new_state, done)
                u, v, w, x, y, z, p, q, r, psi, theta, phi = env.state
                pbar.set_postfix(R='{:.2f}'.format(episode_reward),
                                 w='{:.2f}'.format(w), v='{:.2f}'.format(v),
                                 u='{:.2f}'.format(u),
                                 p='{:.2f}'.format(p), q='{:2f}'.format(q),
                                 r='{:.2f}'.format(r),
                                 psi='{:.2f}'.format(rem(psi, TAU)),
                                 theta='{:.2f}'.format(rem(theta, TAU)),
                                 phi='{:.2f}'.format(rem(phi, TAU)),
                                 z='{:.2f}'.format(z), y='{:.2f}'.format(y),
                                 x='{:.2f}'.format(x))
                pbar.update(1)
                if len(agent.memory) > BATCH_SIZE:
                    agent.update(BATCH_SIZE)
                if done:
                    break
                state = new_state
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    return rewards, avg_rewards


if __name__ == "__main__":
    # mpl.style.use('seaborn')
    plt.style.use("fivethirtyeight")

    PATH = 'results_ddpg/' + date_as_path + '/'
    if not SHOW:
        pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)

    env = AgentEnv(QuadcopterEnv())
    agent = DDPGagent(env)
    rewards, avg_rewards = train(agent, env)
    env.noise_on = False
    agent.save(PATH)
    smth_rewards = smooth(rewards, 30)

    plot_performance(smth_rewards, avg_rewards, rewards,
                     xlabel='episodes', ylabel=r'$r_t(\tau)$',
                     title='Entrenamiento',
                     labels=['smooth reward',
                             'average reward', 'episode reward']
                     )
    # plt.plot(rewards, 'lightskyblue', label='episode reward', alpha=0.05)
    # plt.plot(smooth(rewards, 30), 'blue', label='smooth reward', alpha=0.5)
    # plt.plot(avg_rewards, 'royalblue', label='average reward', alpha=0.01)
    # plt.xlabel('episodes')
    # plt.title('Training - Cumulative Reward $R$')
    # plt.legend(loc='best')
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + 'train_rewards.png')  # , bbox_inches='tight')
        plt.close()

    n_states, n_actions, n_scores = nSim(agent, env, n)
    fig1, _ = plot_nSim2D(n_states, STATE_NAMES, env.time)
    fig2, _ = plot_nSim2D(n_actions, ACTION_NAMES, env.time)
    fig3, _ = plot_nSim2D(n_scores, SCORE_NAMES, env.time)
    fig4, _ = plot_nSim3D(n_states)
    if SHOW:
        plt.show()
    else:
        fig1.savefig(PATH + 'state_rollouts.png')
        fig2.savefig(PATH + 'action_rollouts.png')
        fig3.savefig(PATH + 'score_rollouts.png')
        fig4.savefig(PATH + 'flight_rollouts.png')
    if not SHOW:
        create_report_ddpg(PATH)
        send_correo(PATH + 'Reporte.pdf')

        with open(PATH + 'train_rewards.npy', 'wb') as f:
            np.save(f, np.array(rewards))
