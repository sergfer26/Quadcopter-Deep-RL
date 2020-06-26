import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from quadcopter_env_z import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, control_feedback
from time import time 
from utils import NormalizedEnv, OUNoise
from ddpg import DDPGagent
from time import time
from numpy.linalg import norm
from tqdm import tqdm 

BATCH_SIZE = 32

env = QuadcopterEnv()
env = NormalizedEnv(env)

c1 = (((2*K)/M) * omega_0)**(-1)
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T


if len(sys.argv) == 1:
    hidden_sizes = [64,64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]
    
agent = DDPGagent(env,hidden_sizes)
noise = OUNoise(env.action_space)

writer_train = SummaryWriter()
writer_test = SummaryWriter()


def get_score(state):
    w, z = state
    if ZE - 0.20 < z < ZE + 0.20 and abs(w) < 0.25:
        return 1
    else:
        return 0


def training_loop(agent, env, noise, pbar, test=False):
    state = env.reset()
    if not test:
        noise.reset()
    episode_reward = 0
    score = 0
    s = 0
    while True:
        action = agent.get_action(state)
        if not test:
            action = noise.get_action(action, env.time[env.i])
        control = action * np.ones(4) + W0
        new_state, reward, done = env.step(control)
        score += get_score(new_state)
        episode_reward += reward
        if not test:
            agent.memory.push(state, action, reward, new_state, done)
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)
        z, w = new_state
        pbar.set_postfix(reward='{:.2f}'.format(episode_reward), z='{:.2f}'.format(z), w='{:.2f}'.format(w), sigma='{:.3f}'.format(noise.sigma))
        pbar.update(1)
        if done:
            break
        state = new_state
        s += 1
    return score / s


def train(agent, env, noise, episodes, writer_train, writer_test):
    train_time = 0.0
    max_sigma = ((16.6/ noise.decay_period) * noise.min_sigma)/( 16.6 /noise.decay_period -1)
    sigmas = np.linspace(noise.sigma, max_sigma, episodes)
    for episode in range(episodes):
        start_time = time()
        noise.max_sigma = sigmas[episode]
        with tqdm(total = STEPS, position=0) as pbar_train:
            pbar_train.set_description(f'Episode {episode + 1}/'+str(episodes)+' - training')
            pbar_train.set_postfix(reward='0.0', w='0.0', z='0.0', sigma='0.0')
            train_score = training_loop(agent, env, noise, pbar_train)
            train_time +=  time() - start_time
            writer_train.add_scalar('episode vs score', train_score, episode)
        with tqdm(total = STEPS, position=0) as pbar_test:
            pbar_test.set_description(f'Episode {episode + 1}/'+str(episodes)+' - test')
            pbar_test.set_postfix(reward='0.0', w='0.0', z='0.0')
            test_score = training_loop(agent, env, noise, pbar_test, test=True)
            writer_train.add_scalar('episode vs score', test_score, episode)

def Sim(flag, agent, env):
    t = env.time
    fig, ((ax1, ax2)) = plt.subplots(2, 1)
    state = env.reset()
    noise.reset()
    episode_reward = 0
    R = []
    W1,W2 = [],[]
    Z1,Z2 = [] ,[]
    A = []
    env.flag  = flag
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        control = action*np.ones(4) + W0
        new_state, reward, done = env.step(control) 
        z,w = state
        W1.append(w)
        Z1.append(z)
        R.append(reward)
        A.append(action)
        state = new_state
        episode_reward += reward
        if done:
            break
    t1 = t[0:len(Z1)]
    ax1.set_ylim(-10, 30)
    ax1.plot(t1,Z1,'-r',label = str(round(Z1[-1],4)))
    ax1.plot(t1,W1,'--b',label = str(round(W1[-1],4)))
    ax1.plot(t1,15*np.ones(len(t1)), '--')
    state = env.reset()
    env.i = 0
    while True:
        W1 = control_feedback(env.state[0]-env.z_e, env.state[1], F1) * c1
        control = W1 + W0
        new_state, reward, done = env.step(control)
        z,w = state
        W2.append(w)
        Z2.append(z)
        state = new_state
        if done:
            break
    t2 =  t[0:len(Z2)]
    ax1.legend()
    ax2.plot(t2,Z2,'-r',label = str(round(Z2[-1],4)))
    ax2.plot(t2,W2,'--b',label = str(round(W2[-1],4)))
    ax2.legend()
    ax1.title.set_text('DDPG')
    ax2.title.set_text('Control Lineal')
    plt.show()

train(agent, env, noise, 200, writer_train, writer_test)
Sim(True, agent, env)

hs = hidden_sizes
name = ''
for s in hs:
    name += '_'+str(s)
torch.save(agent.actor.state_dict(), "ddpg/saved_models/actor"+name+"_.pth")
torch.save(agent.critic.state_dict(), "ddpg/saved_models/critic"+name+"_.pth")