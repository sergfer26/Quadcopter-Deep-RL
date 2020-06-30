

#!/usr/bin/env python
import numpy as np
import gym
from gym import spaces
from numpy import pi
#!/usr/bin/env python3
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from numpy import tanh
from numpy import sqrt
from numpy import floor
from numpy.linalg import norm
from torch.utils.tensorboard import SummaryWriter
import sys
import matplotlib.pyplot as plt
from time import time
r2 = lambda z, w : (1 - tanh(z)) *w**2
rt = lambda t : max(0,t - 15)




# ## Sistema dinámico

# constantes
G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))


def f(y, t, w1, w2, w3, w4):
    #El primer parametro es un vector
    #W tambien
    z,w = y
    dz = w
    W = [w1, w2 , w3 , w4]
    dw = G - (K/M) * norm(W) ** 2
    #dw = (-2*K/M)*omega_0*(w1 + w2 + w3 + w4)
    return dz, dw


# def f(y, t, w1, w2, w3, w4):
#     #El primer parametro es un vector
#     #W,I tambien
#     u, v, w, p, q, r, psi, theta, phi, x, y, z = y
#     Ixx, Iyy, Izz = I
#     W = np.array([w1, w2, w3, w4])
#     du = r * v - q * w - G * sin(theta)
#     dv = p * w - r * u - G * cos(theta) * sin(phi)
#     dw = q * u - p * v + G * cos(phi) * cos(theta) - (K/M) * norm(W) ** 2
#     dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
#     dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
#     dr = (B / Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
#     dpsi = (q * sin(phi) + r * cos(phi)) * (1 / cos(theta))
#     dtheta = q * cos(phi) - r * sin(phi)
#     dphi = p + (q * sin(phi) + r * cos(phi)) * tan(theta)
#     dx = u
#     dy = v
#     dz = w
#     return du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz
#     # return dz, dw

# ## Diseño del ambiente, step y reward


# constantes del ambiente
Vel_Max = 60 #Velocidad maxima de los motores
Vel_Min = -20
Low_obs = np.array([0,-10,0])#z,w,t
High_obs = np.array([18,10,30])

#Tiempo_max = 15
#tam = 


# In[5]:


"""Quadcopter Environment that follows gym interface"""

class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box(low = Vel_Min * np.ones(1), high = Vel_Max*np.ones(1))
        self.observation_space = spaces.Box(low = Low_obs, high = High_obs)
        self.i = 0
        self.time_max = 20
        self.tam = 590
        self.time = np.linspace(0, self.time_max, self.tam)
        self.rz, self.rw = [1,0.5]
        self.state = self.reset()
        self.z_e = 15
        self.flag = True
        
    def reward_f(self):
        z,w,t = self.state
        if z <= 0:
            return -1e2
        #else:
        #return -np.linalg.norm([z-self.z_e,2*w])
            #return -0.5*(abs(z-self.z_e) + abs(w))
        #return -10e-4*sqrt((z-self.z_e)**2 + w**2 )
        #return np.tanh(1 - 0.00005*(abs(self.state - np.array([15,0]))).sum()
        #if 13 < z < 17:
        #    import pdb; pdb.set_trace()
            
        #else:
        #    return 0
        else:
            return -np.linalg.norm([10*(z-self.z_e), r2(abs(z - self.z_e), 2*w),rt(t)])
            #return - abs(z-self.z_e) - r2(abs(z - self.z_e), w)
            #return -np.linalg.norm([z-self.z_e,w**2])
            #return -((z-self.z_e)**2 + r2(z-self.z_e, w))
            
        
    def is_done(self):
        z,w,_ = self.state
        #Si se te acabo el tiempo
        if self.i == self.tam-2:
            return True
        #Si estas muy lejos
        #elif self.reward_f() < -1e3:
        #    return True
        #Si estas muy cerca
        #elif self.reward_f() > -1e-3:
        #    return True
        elif self.flag:
            if z < 0 or z > 18:
                return True
            else:
                return False
            #elif 0 < w < High_obs[1]:
           #  return True
          ##else:
            # return False

    def step(self,action):
        #import pdb; pdb.set_trace()
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f, self.state[0:2], t, args=(w1, w2, w3, w4))[1]
        self.state = [delta_y[0],delta_y[1],self.time[self.i+1]]
        reward = self.reward_f()
        done = self.is_done()
        self.i += 1
        return np.array([float(delta_y[0]),float(delta_y[1]),self.time[self.i]]), reward, done

    def reset(self):
        z0 = max(0,15 + float(np.random.uniform(-self.rz,self.rz,1)))
        w0 = float(np.random.uniform(-self.rw,self.rw,1))
        self.i = 0
        t0 = self.time[self.i]
        self.state = np.array([z0,w0,t0])
        #self.state = np.array([10 ,0])
        #self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
        return self.state
        
    def render(self, mode='human', close=False):
        pass

# In[6]:


def control_feedback(x, y, F):
    '''
    Realiza el control lineal de las velocidades W
    dadas las variables (x, y).
    param x: variable independiente (dx = y)
    param y: variable dependiente
    param F: matriz 2x4 de control
    regresa: W = w1, w2, w3, w4 
    '''
    A = np.array([x, y]).reshape((2, 1))
    return np.dot(F, A).reshape((4,))


# ## Proceso Ornstein-Ulhenbeck para la exploración
# 
# $$\mu(s_t) = \mu(s_t | \theta_t^{\mu}) + \mathcal{N}$$

# In[7]:


import numpy as np
import gym
from collections import deque
import random

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.8, min_sigma=0.2, decay_period=100000):

        
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        #print(action)
        #print(action + ou_state)
        return np.clip(action + ou_state, self.low, self.high)


class NormalizedEnv(QuadcopterEnv): #(gym.ActionWrapper):
    """ Wrap action """ #Recibe un ambiente
    def __init__(self, env):
        QuadcopterEnv.__init__(self)
        
    
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    


# ## Replay Buffer, almacenamiento de la transición

# In[8]:


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    def remove(self):
        n = int(floor(len(self.buffer)/2))
        for _ in range(n):
            i = np.random.randint(n)
            del self.buffer[i]
        print('Se elimino el 50% de la infomacion!!')


# # Diseño de arquitecturas para DDPG
# Parámetros: 
# $$
# \begin{align*}
# \theta^Q &: \text{ Q network} \\
# \theta^{\mu}&: \text{ Deterministic policy function} \\
# \theta^{Q'} &: \text{ target Q network} \\
# \theta^{\mu '}&: \text{ target policy function}
# \end{align*}
# $$

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
    

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


# # Diseño del agente y función de entrenamiento

# In[10]:


import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
#from model import *
#from utils import *

class DDPGagent:
    def __init__(self, env, hidden_size=64, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=10000000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        #lambda2 = torch.tensor(1.)
        #l2_reg = torch.tensor(0.)
        #for param in self.critic.parameters():
            #l2_reg += torch.norm(param)
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean() #+ lambda2 * l2_reg
        #import pdb;pdb.set_trace()
        
        
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))






# ## Prueba de control en z y w

# In[11]:


#env = NormalizedEnv(gym.make("Pendulum-v0"))


env = QuadcopterEnv()
env = NormalizedEnv(env)
agent = DDPGagent(env)
noise = OUNoise(env.action_space)





omega_0 = np.sqrt((G * M)/(4 * K))
c1 = (((2*K)/M) * omega_0)**(-1)
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T



def train(episodios,rz,rw):
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
            print(state)
            action = agent.get_action(state)
            action = noise.get_action(action, env.time[env.i])
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
         #if episode > 240:
             #t = t[0:len(Z)]
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

train(1,0,0)
#train(100,2,0.3)
#train(100,3,0.5)
#train(100,3,0.7)
#train(150,4,1.0)
#train(150,5,1.3)
#train(200,6,1.5)


def reset_time(tamaño,tiempo_max):
    env.time_max = tiempo_max
    env.tam = tamaño
    env.time = np.linspace(0, env.time_max, env.tam)


reset_time(882,30)

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
        z,w,__ = state
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
        z,w,_= state
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
    return norm([state[0]-15,state[1]])

def ntest(n):
    suma = 0
    for _ in range(n):
        suma += test()
    return suma/n

