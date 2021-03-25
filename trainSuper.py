import numpy as np
import pandas as pd
import torch
import pytz
import pathlib
import gym
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from quadcopter_env import QuadcopterEnv, AgentEnv
from DDPG.ddpg import DDPGagent
from Linear.step import control_feedback, omega_0, C, F
from trainDDPG import sim
from progress.bar import Bar
from tqdm import tqdm
from params import PARAMS_TRAIN_SUPER
from graphics import *
from get_report import*

plt.style.use('ggplot')

c1, c2, c3, c4 = C
F1, F2, F3, F4 = F
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
BATCH_SIZE = PARAMS_TRAIN_SUPER['BATCH_SIZE']
EPOCHS = PARAMS_TRAIN_SUPER['EPOCHS']
N = PARAMS_TRAIN_SUPER['N']
LOW_OBS = np.array([-0.05, -0.05, -0.05,  -5, -5, -5, 0.05,
                   0.05, 0.05, -np.pi/32, -np.pi/32, -np.pi/32])
HIGH_OBS = np.array([0.05, 0.05, 0.05, 5, 5, 5, 0.05, 0.05,
                    0.05, np.pi/32, np.pi/32, np.pi/32])

DEVICE = "cpu"
DTYPE = torch.float
SHOW = PARAMS_TRAIN_SUPER['SHOW']

env = QuadcopterEnv()
env.observation_space = gym.spaces.Box(low=LOW_OBS, high=HIGH_OBS,dtype=np.float64)
env = AgentEnv(env)
agent = DDPGagent(env)
env.noise_on = False


if torch.cuda.is_available():
    DEVICE = "cuda"
    agent.actor.cuda()  # para usar el gpu
    agent.critic.cuda()



class Memory_Dataset(Dataset):
    def __init__(self, buffer, env):
        K = len(buffer)
        self.states = torch.zeros(
            K, int(env.observation_space.shape[0]), dtype=DTYPE, device=DEVICE)
        self.actions = torch.zeros(
            K, int(env.action_space.shape[0]), dtype=DTYPE, device=DEVICE)
        self.rewards = torch.zeros(K, 1, dtype=DTYPE, device=DEVICE)
        self.next_states = torch.zeros(
            K, int(env.observation_space.shape[0]), dtype=DTYPE, device=DEVICE)
        for k in range(K):
            s, a, r, ns, _ = buffer[k]
            self.states[k, :] = torch.tensor(s, dtype=DTYPE, device=DEVICE)
            self.actions[k, :] = torch.tensor(a, dtype=DTYPE, device=DEVICE)
            self.rewards[k, :] = torch.tensor(r, dtype=DTYPE, device=DEVICE)
            self.next_states[k, :] = torch.tensor(
                ns, dtype=DTYPE, device=DEVICE)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]


def get_action(state, goal):
    _, _, _, _, _, z_e, _, _, _, psi_e, theta_e, phi_e = goal
    _, _, w, _, _, z, p, q, r, psi, theta, phi = state
    W1 = control_feedback(z - z_e, w, F1 * c1).reshape(1, 4)[0]  # control z
    W2 = control_feedback(psi - psi_e, r, F2 *
                          c2).reshape(1, 4)[0]  # control yaw
    W3 = control_feedback(phi - phi_e, p, F3 *
                          c3).reshape(1, 4)[0]  # control roll
    W4 = control_feedback(theta - theta_e, q, F4 *
                          c4).reshape(1, 4)[0]  # control pitch
    W = W1 + W2 + W3 + W4
    return W


def get_experience(env, memory, n):
    #print('Learning from observations')
    goal = env.goal
    #bar = Bar('Processing', max=n)
    k = 0
    for _ in range(n):
        #bar.next()
        state = env.reset()
        for _ in range(env.steps):
            real_action = get_action(env.state, goal)
            action = env.reverse_action(real_action)
            action += np.random.normal(0,0.1,4)
            _, reward, new_state, done = env.step(action)
            new_state[0:9] += np.random.normal(0,1,9)
            #if (abs(real_action) < env.action_space.high[0]).all():
            memory.push(state, action, reward, new_state, done)

            state = new_state

            if done:
                break
    #bar.finish()
    #print(k)


def train(agent, env, data_loader):
    n_batches = len(data_loader)
    Scores = {'$Cr_t$': list(), 'stable': list(), 'contained': list()}
    Loss = {'policy': np.zeros(EPOCHS), 'critic': np.zeros(EPOCHS)}
    for epoch in range(1, EPOCHS + 1):
        #with tqdm(total=len(agent.memory.buffer), position=0) as pbar:
            #pbar.set_description(f'Epoch {epoch}/' + str(EPOCHS))
        for i, data in enumerate(data_loader, 0):
            states, actions, rewards, next_states = data
            policy_loss, critic_loss = agent.train(
                states, actions, rewards, next_states)
            Loss['policy'][epoch - 1] += policy_loss / len(data)
            Loss['critic'][epoch - 1] += critic_loss / len(data)
            #pbar.set_postfix(policy_loss='{:.4f}'.format(
                #policy_loss), critic_loss='{:.4f}'.format(critic_loss))
            #pbar.update(states.shape[0])

        if epoch % 10 == 0:
            # r_t, Cr_t, stable, contained
            states, actions, scores = sim(True, agent, env)
            Scores['$Cr_t$'].append(scores[1, -1])
            Scores['stable'].append(np.sum(scores[2, :]))
            Scores['contained'].append(np.sum(scores[3, :]))

    Loss['policy'] /= n_batches
    Loss['critic'] /= n_batches
    return Loss, Scores


if __name__ == "__main__":
    
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute

    PATH = 'results_super/' + str(month) + '_' + \
        str(day) + '_' + str(hour) + str(minute)
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)

    get_experience(env, agent.memory, N)

    dataset = Memory_Dataset(agent.memory.buffer, env)
    n_samples = len(agent.memory.buffer)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    Loss, Scores = train(agent, env, data_loader)
    agent.save(PATH)

    data_loss = pd.DataFrame(Loss, columns=list(Loss.keys()))
    data_scores = pd.DataFrame(Scores, columns=list(Scores.keys()))

    data_loss.plot(subplots=True, layout=(1, 2),
                   figsize=(10, 7), title='Training loss')

    plt.title(r'$\tau =$' + '{}'.format(agent.tau))
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/loss.png')

    data_scores.plot(subplots=True, layout=(
        2, 2), figsize=(10, 7), title='Validation scores')
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/validation_scores.png')
    #env.set_time(2400,90)
    nsim3D(5, agent, env,PATH)
    nsim2D(5, agent, env, PATH)
    create_report_super(PATH) 
