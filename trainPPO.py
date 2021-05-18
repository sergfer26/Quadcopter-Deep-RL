import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import pytz
import pathlib
from quadcopter_env import QuadcopterEnv, AgentEnv
from PPO.ppo import PPOagent
from datetime import datetime
from get_report import create_report_ppo
from params import PARAMS_TRAIN_PPO
from simulation import nSim, plot_nSim2D, plot_nSim3D


SHOW = True  # PARAMS_TRAIN_PPO['SHOW']
EPISODES = PARAMS_TRAIN_PPO['EPISODES']
action_std_decay_freq = PARAMS_TRAIN_PPO['action_std_decay_freq']


def train(agent, env, action_std_decay_freq):
    update_timestep = env.steps
    time_step = 0
    rewards = []
    avg_rewards = []
    for _ in range(EPISODES):
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
    plt.title('Training - Cumulative Reward')
<<<<<<< HEAD
    # r'$r_t = \mathbb{1}_{x <= g + 1} - 0.01 \|x - g\| - 0.01 \|[dx, d\theta]\|
    # - 0.5 \|I - X_{\theta}\|$')
=======
    # r'$r_t = \mathbb{1}_{x <= g + 1} - 0.01 \|x - g\| - 0.01 \|[dx, d\theta]\| - 0.5 \|I - X_{\theta}\|$')
>>>>>>> 23d2da0f4d984469e1d29274d3270369651f9aa3
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/c_rewards.png', bbox_inches='tight')
        plt.close()

    n_states, n_actions, n_scores = nSim(False, agent, env, 10)
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
