import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from env.quadcopter_env_z import QuadcopterEnv, G, M, K, omega_0, control_feedback, STEPS
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
    if 14.9 < z < 15.1 and abs(w) < 0.0:
        return 1
    else:
        return 0

def training_loop(agent, noise, pbar, test=False):
    state = agent.env.reset()
    noise.reset()
    episode_reward = 0
    score = 0.00
    s = 0
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        control = action * np.ones(4) + W0
        new_state, reward, done = env.step(control)
        score += get_score(new_state)
        episode_reward += reward
        if not test:
            agent.memory.push(state, action, reward, new_state, done)
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)
        if s % 10 == 0:
            z, w = new_state
            pbar.set_postfix(reward='{:.2f}'.format(episode_reward), z='{:.2f}'.format(z), w='{:.2f}'.format(w))
            pbar.update(10)
        if done:
            break
        state = new_state
        s += 1
    return score / s

def train_episode(agent, noise, episodes, writer_train, writer_test):
    for episode in range(episodes):
        start_time = time()
        with tqdm(total = STEPS, position=0) as pbar_train:
            pbar_train.set_description(f'Episode {episode + 1}/'+str(episodes)+' - training')
            pbar_train.set_postfix(reward='0.0', w='0.0', z='0.0')
            train_score = training_loop(agent, noise, pbar_train)
            train_time +=  time() - start_time
            writer_train.add_scalar('episode vs train_score', episode, train_score)
        with tqdm(total = STEPS, position=0) as pbar_test:
            pbar_test.set_description(f'Episode {episode + 1}/'+str(episodes)+' - test')
            pbar_test.set_postfix(reward='0.0', w='0.0', z='0.0')
            test_score = training_loop(agent, noise, pbar_test, test=True)
            writer_train.add_scalar('episode vs test_score', episode, test_score)


def train(episodios, rz, rw):
    tic = time()
    env.rz = rz
    env.rw = rw
    rewards = []
    avg_rewards = []
    batch_size = 32
    writer = SummaryWriter()
    t = env.time
    #fig, ((ax1, ax2)) = plt.subplots(2, 1)
    for episode in range(episodios):
        state = env.reset()
        noise.reset()
        episode_reward = 0
         #R = []
         #W = []
         #Z = []
         #t = env.time
         #A = []
        while True:
            action = agent.get_action(state)
            action = noise.get_action(action, env.time[env.i])
            print(noise.sigma)
            control = action*np.ones(4) + W0
            new_state, reward, done = env.step(control) 
            agent.memory.push(state, action, reward, new_state, done)
            if len(agent.memory) > batch_size :
                agent.update(batch_size)
                if episode > 240:
                    pass
                     #z,w = state
                     #W.append(w)
                     #Z.append(z)
                     #R.append(reward)
            state = new_state
            episode_reward += reward
            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.round(np.mean(rewards[-10:])),4) )
                print(state,round(reward,4),round(env.time[env.i],2))
            #writer.add_scalar('episodio vs Z', state[0], episode)
                writer.add_scalars('episodio vs Z', {'Z':state[0] ,'z_e':15},episode)
                writer.add_scalar('episodio vs W', state[1], episode)
                writer.add_scalar('episodio vs Tiempo',env.time[env.i],episode)
                writer.add_scalar('episodio vs Reward', reward, episode)
                break
    #writer.add_scalar('Episode vs  Episode_Reward', episode, episode_reward)
        if episode > 240:
            t = t[0:len(Z)]
             #ax1.set_ylim(-10, 30)
             #ax1.plot(t,Z,'-r',t,W,'--b',t,R,alpha = 0.2)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

 #ax2.plot(rewards)
 #ax2.plot(avg_rewards)
#plt.xlabel('Episode')
#plt.ylabel('Reward')
#plt.show()
    agent.memory.remove()
    toc = time()
    print('Tiempo de Ejecucion = ',(toc-tic)/60.0,' minutos')
 #plt.show()



def reset_time(tamaño,tiempo_max):
    env.time_max = tiempo_max
    env.tam = tamaño
    env.time = np.linspace(0, env.time_max, env.tam)


#reset_time(882,30)

def Sim(flag):
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
        state = state 
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
        W1 = control_feedback(env.state[0]-env.ze, env.state[1], F1) * c1
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


train(200,1,0)
Sim(True)
#train(100,2,0.3)
#train(100,4,0.5)
#train(100,3,0.7)
#train(100,4,1.0)
#train(150,5,1.3)
#train(200,6,1.5)ç

[100, 100, 100, 150, 200]


def test():
    state = env.reset()
    noise.reset()
    episode_reward = 0
    while True:
        state = state 
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        control = action*np.ones(4) + W0
        new_state, reward, done = env.step(control) 
        state = new_state
        episode_reward += reward
        if done:
            break
    return state[0]-15,state[1]

def ntest(n):
    final_z = []
    final_w = []
    for _ in range(n):
        z,w = test()
        final_z.append(z)
        final_w.append(w)
    return final_z,final_w

def hist(z,w,dim):
    if dim == 1:
        fig, ((ax1, ax2)) = plt.subplots(1, 2)
        ax1.hist(z,label = 'Z', color = 'navy')
        ax2.hist(w,label = 'W')
        fig.suptitle(' beta  = ' + str(env.beta) + ', ' +'epsilon = ' + str(env.epsilon) , fontsize=16)
        ax1.legend()
        ax2.legend()
    elif dim == 2:
        plt.hist2d(z, w, bins=(50, 50), cmap=plt.cm.jet)
        plt.xlabel('Z')
        plt.ylabel('W')
    plt.show()



    

#env.rz = 6
#env.rw = 1.2
#final_z,final_w = ntest(500)
#hist(final_z,final_w,2)



