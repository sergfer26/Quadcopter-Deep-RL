import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import pytz
import pathlib
from tqdm import tqdm
from numpy import remainder as rem
from env import QuadcopterEnv, AgentEnv
from PPO.ppo import PPOagent
from datetime import datetime
from get_report import create_report_ppo
from params import PARAMS_TRAIN_PPO
from simulation import nSim, plot_nSim2D, plot_nSim3D


SHOW = PARAMS_TRAIN_PPO['SHOW']
EPISODES = PARAMS_TRAIN_PPO['EPISODES']
TAU = 2 * np.pi  # No es el tau del agente
n = PARAMS_TRAIN_PPO['n']
action_std_decay_freq = PARAMS_TRAIN_PPO['action_std_decay_freq']

if not SHOW:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def train(agent, env, action_std_decay_freq):
    update_timestep = env.steps
    time_step = 0
    rewards = []
    avg_rewards = []
    for episode in range(EPISODES):
        with tqdm(total=env.steps) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
            state = env.reset()
            episode_reward = 0
            while True:
                action = agent.get_action(state)
                _, reward, new_state, done = env.step(action)

                # saving reward and is_terminals
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)

                time_step += 1
                episode_reward += reward

                if time_step % update_timestep == 0:
                    agent.update()

                if time_step % action_std_decay_freq == 0:
                    agent.decay_action_std()

                if done:
                    agent.buffer.clear()
                    break
                state = new_state
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
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
    return rewards, avg_rewards


if __name__ == "__main__":
    # tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now()  # tz
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute

    mpl.style.use('seaborn')

    PATH = 'results_ppo/' + str(month) + '_' + \
        str(day) + '_' + str(hour) + str(minute)
    if not SHOW:
        pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)

    env = AgentEnv(QuadcopterEnv())
    env.noise_on = False
    agent = PPOagent(env)
    rewards, avg_rewards = train(agent, env, action_std_decay_freq)
    agent.save(PATH)
    plt.plot(rewards, 'b--', label='episode reward', alpha=0.1)
    plt.plot(avg_rewards, 'r-', label='average reward')
    plt.xlabel('episodes')
    plt.title('Training - Cumulative Reward - ' +
              r'$\lambda = {}$'.format(env.lamb))
    # r'$r_t = \mathbb{1}_{x <= g + 1} - 0.01 \|x - g\| - 0.01 \|[dx, d\theta]\|
    # - 0.5 \|I - X_{\theta}\|$')
    # r'$r_t = \mathbb{1}_{x <= g + 1} - 0.01 \|x - g\| - 0.01 \|[dx, d\theta]\| - 0.5 \|I - X_{\theta}\|$')
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/c_rewards.png', bbox_inches='tight')
        plt.close()

    n_states, n_actions, n_scores = nSim(False, agent, env, n)
    columns = ('$u$', '$v$', '$w$', '$x$', '$y$', '$z$', '$p$',
               '$q$', '$r$', r'$\psi$', r'$\theta$', r'$\varphi$')
    plot_nSim2D(n_states, columns, env.time, show=SHOW,
                file_name=PATH + '/sim_states.png')
    columns = ['$a_{}$'.format(i) for i in range(1, 5)]
    plot_nSim2D(n_actions, columns, env.time, show=SHOW,
                file_name=PATH + '/sim_actions.png')
    columns = ('$r_t$', '$Cr_t$', 'is stable', 'cotained')
    plot_nSim2D(n_scores, columns, env.time, show=SHOW,
                file_name=PATH + '/sim_scores.png')
    plot_nSim3D(n_states, show=SHOW, file_name=PATH + '/sim_flights.png')
    if not SHOW:
        create_report_ppo(PATH)
        with open(PATH + '/training_rewards.npy', 'wb') as f:
            np.save(f, np.array(rewards))
        with open(PATH + '/training_avg_rewards.npy', 'wb') as f:
            np.save(f, np.array(avg_rewards))
