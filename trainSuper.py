import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from quadcopter_env import QuadcopterEnv, AgentEnv
from DDPG.ddpg import DDPGagent 
from Linear.step import control_feedback, omega_0, C, F
from trainDDPG import sim
from progress.bar import Bar
from tqdm import tqdm

plt.style.use('ggplot')

c1, c2, c3, c4 = C
F1, F2, F3, F4 = F
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
BATCH_SIZE = 32
EPOCHS = 500
N = 1000 # vuelos simulados

DEVICE = "cpu"
DTYPE = torch.float32

env = AgentEnv(QuadcopterEnv())
agent = DDPGagent(env)
agent.noise_on = False
agent.tau = 1.0
if torch.cuda.is_available(): 
    DEVICE = "cuda"
    agent.actor.cuda() # para usar el gpu
    agent.critic.cuda()


class Memory_Dataset(Dataset):
    
    def __init__(self, buffer, env):
        K = len(buffer)
        self.states = torch.zeros(K, int(env.observation_space.shape[0]), dtype=DTYPE, device=DEVICE)
        self.actions = torch.zeros(K, int(env.action_space.shape[0]), dtype=DTYPE, device=DEVICE)
        self.rewards = torch.zeros(K, 1, dtype=DTYPE, device=DEVICE)
        self.next_states = torch.zeros(K, int(env.observation_space.shape[0]), dtype=DTYPE, device=DEVICE)

        for k in range(K):
            s, a, r, ns, _ = buffer[k]
            self.states[k, :] = torch.tensor(s, dtype=DTYPE, device=DEVICE)
            self.actions[k, :] = torch.tensor(a, dtype=DTYPE, device=DEVICE)
            self.rewards[k, :] = torch.tensor(r, dtype=DTYPE, device=DEVICE)
            self.next_states[k, :] = torch.tensor(ns, dtype=DTYPE, device=DEVICE)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]


def get_action(state, goal):
    _, _, _, _, _, z_e , _, _, _, psi_e, theta_e, phi_e = goal
    _, _, w, _, _, z, p, q, r, psi, theta, phi = state
    W1 = control_feedback(z - z_e, w, F1 *c1).reshape(1, 4)[0]  # control z
    W2 = control_feedback(psi - psi_e, r, F2 * c2).reshape(1, 4)[0]  # control yaw
    W3 = control_feedback(phi - phi_e, p, F3 * c3).reshape(1, 4)[0]  # control roll
    W4 = control_feedback(theta - theta_e, q, F4 * c4).reshape(1, 4)[0]  # control pitch
    W = W1 + W2 + W3 + W4
    return W


def get_experience(env, memory, n):
    print('Learning from observations')
    goal = env.goal
    bar = Bar('Processing', max=n)
    for _ in range(n):
        bar.next()
        state = env.reset()
        for _ in range(env.steps):
            obs = env.reverse_observation(state)
            real_action = get_action(obs, goal)
            action = env.reverse_action(real_action)
            new_state, reward, done = env.step(action)
            memory.push(state, action, reward, new_state, done)
            state = new_state
            if done:
                break
    bar.finish()


def train(agent, env, data_loader):
    n_batches = len(data_loader)
    Scores = {'$Cr_t$': list(), 'stable': list(), 'contained': list()}
    Loss = {'policy': np.zeros(EPOCHS), 'critic': np.zeros(EPOCHS)}
    for epoch in range(1, EPOCHS + 1):
        with tqdm(total = len(agent.memory.buffer), position=0) as pbar:
            pbar.set_description(f'Epoch {epoch}/' + str(EPOCHS))
            for i, data in enumerate(data_loader, 0):
                states, actions, rewards, next_states = data
                policy_loss, critic_loss = agent.train(states, actions, rewards, next_states)
                Loss['policy'][epoch -1] += policy_loss / len(data)
                Loss['critic'][epoch- 1] += critic_loss / len(data)
                pbar.set_postfix(policy_loss='{:.4f}'.format(policy_loss), critic_loss='{:.4f}'.format(critic_loss))
                pbar.update(states.shape[0])

            if epoch % 10 == 0:
                _, _, scores = sim(True, agent, env) # r_t, Cr_t, stable, contained
                Scores['$Cr_t$'].append(scores[1, -1])
                Scores['stable'].append(np.sum(scores[2, :]))
                Scores['contained'].append(np.sum(scores[3, :]))

    Loss['policy'] /= n_batches
    Loss['critic'] /= n_batches
    return Loss, Scores
  

get_experience(env, agent.memory, N)
dataset = Memory_Dataset(agent.memory.buffer, env)
n_samples = len(agent.memory.buffer)
data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
Loss, Scores = train(agent, env, data_loader)

data_loss = pd.DataFrame(Loss, columns=list(Loss.keys()))
data_scores = pd.DataFrame(Scores, columns=list(Scores.keys()))


data_loss.plot(subplots=True, layout=(1, 2), figsize=(10, 7), title='Training loss')
plt.show()

data_scores.plot(subplots=True, layout=(2, 2), figsize=(10, 7), title='Validation scores')
plt.show()








