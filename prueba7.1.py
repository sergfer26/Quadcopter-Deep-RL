import numpy as np
import pandas as pd
import sys
import torch
from DDPG.env.quadcopter_env import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, control_feedback,funcion
from time import time 
from DDPG.utils import NormalizedEnv, OUNoise
from DDPG.ddpg import DDPGagent
from time import time
from numpy.linalg import norm
from tqdm import tqdm
from DDPG.load_save import load_nets, save_nets
from tools.tools import imagen2d,imagen
from numpy import pi,cos,sin
from numpy import remainder as rem
from progress.bar import Bar, ChargingBar
#from vpython import*
TAU = 2 * pi
BATCH_SIZE = 32
env = QuadcopterEnv()
env = NormalizedEnv(env)
c1 = (((2*K)/M) * omega_0)**(-1)
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T

vector_u, vector_v, vector_w = [],[],[]
vector_p, vector_q, vector_r = [],[],[]
vector_psi, vector_theta, vector_phi = [],[],[]
vector_x, vector_y, vector_z = [],[],[]
vector_score,vector_tiempo,vector_reward,vector_porcentaje = [],[],[],[]
    

def get_score(state):
    z = state[-1]
    w = state[2]
    if abs(z-ZE) < 0.2 and abs(w) < 0.25:
        return 1
    else:
        return 0


def reset_time(env, tamaño, tiempo_max):
    env.time_max = tiempo_max
    env.tam = tamaño
    env.time = np.linspace(0, env.time_max, env.tam)


#u, v, w, p, q, r, psi, theta, phi, x, y, z = env.state
def training_loop(agent, env, noise):
    state = funcion(env.reset())
    noise.reset()
    episode_reward = 0
    score = 0
    s = 1
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        W1 = control_feedback(env.state[11] - 15, env.state[2], F1) * c1  # control z
        control = W0 + action*(1-env.lam) + env.lam*W1
        new_state, reward, done = env.step(control)
        score += get_score(env.state)
        episode_reward += reward
        agent.memory.push(state, action, reward, new_state, done)
        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)
        if done:
            break
        state = new_state
        s += 1
    return score,s,episode_reward


def train(agent, rz, rw, env, noise, episodes):
    train_time = 0.0
    env.p[2], env.p[-1] = rw, rz
    max_sigma = ((16.6 / noise.decay_period) * noise.min_sigma)/( 16.6 /noise.decay_period -1)
    sigmas = np.linspace(noise.sigma, max_sigma, episodes)
    for episode in range(episodes):
        start_time = time()
        noise.max_sigma = max(0,sigmas[episode])
        train_score,s,train_reward= training_loop(agent, env, noise)
        train_score = train_score/s
        train_time +=  time() - start_time
        vector_u.append(env.state[0]);vector_v.append(env.state[1]);vector_w.append(env.state[2])
        vector_p.append(env.state[3]);vector_q.append(env.state[4]);vector_r.append(env.state[5])
        vector_psi.append(env.state[6]);vector_theta.append(env.state[7]);vector_phi.append(env.state[8])
        vector_x.append(env.state[9]);vector_y.append(env.state[10]);vector_z.append(env.state[11])
        vector_score.append(train_score)
        vector_tiempo.append(train_time)
        vector_reward.append(train_reward)
        vector_porcentaje.append(s/(env.tam-1)*100)

def Sim(flag, agent, env):
    t = env.time
    state = funcion(env.reset())
    noise.reset()
    _, _, w, p, q, r, psi, theta, phi, _, _, z = env.state
    episode_reward = 0
    Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
    X,Y = [], []
    env.flag  = flag
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        W1 = control_feedback(env.state[11] - 15, env.state[2], F1) * c1  # control z
        control = W0 + action*(1-env.lam) + env.lam*W1
        new_state, reward, done = env.step(control) 
        _, _, w, p, q, r, psi, theta, phi, x, y, z = env.state
        Z.append(z); W.append(w)
        Psi.append(psi); R.append(r)
        Phi.append(phi); P.append(p)
        Theta.append(theta); Q.append(q)
        X.append(x);Y.append(y)
        state = new_state
        episode_reward += reward
        if done:
            break
    T = t[0:len(Z)]
    imagen2d(Z, W, Psi, R, Phi, P, Theta, Q, T)
    imagen(X,Y,Z)


def nsim(flag,n):
    f, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
    bar1 = Bar('Procesando:', max=n)
    alpha = 0.2
    for _ in range(n):
        bar1.next()
        t = env.time
        state = funcion(env.reset())
        noise.reset()
        _, _, w, p, q, r, psi, theta, phi, _, _, z = env.state
        episode_reward = 0
        Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
        X,Y = [], []
        env.flag  = flag
        while True:
            action = agent.get_action(state)
            action = noise.get_action(action, env.time[env.i])
            W1 = control_feedback(env.state[11] - 15, env.state[2], F1) * c1
            control = W0 + action*(1-env.lam) + env.lam*W1
            new_state, reward, done = env.step(control) 
            _, _, w, p, q, r, psi, theta, phi, x, y, z = env.state
            Z.append(z); W.append(w)
            Psi.append(psi); R.append(r)
            Phi.append(phi); P.append(p)
            Theta.append(theta); Q.append(q)
            X.append(x);Y.append(y)
            state = new_state
            episode_reward += reward
            if done:
                break
        T = t[0:len(Z)]
        cero = np.zeros(len(Z))
        w1.plot(T, Z, c='b',alpha = alpha)
        w1.set_ylabel('z')
        w2.plot(T, W, c='b',alpha = alpha)
        w2.set_ylabel('dz') 
        r1.plot(T, Psi, c='r',alpha = alpha)
        r1.set_ylabel('$\psi$')
        r2.plot(T, R, c='r',alpha = alpha)
        r2.set_ylabel('d$\psi$')
        p1.plot(T, Phi, c='g',alpha = alpha)
        p1.set_ylabel('$\phi$')
        p2.plot(T, P, c='g',alpha = alpha)
        p2.set_ylabel(' d$\phi$',alpha = alpha)
        q1.plot(T, Theta,c = 'k',alpha = alpha)
        q1.set_ylabel('$ \\theta$')
        q2.plot(T, Q,c = 'k' ,alpha = alpha)
        q2.set_ylabel(' d$ \\theta$')
    w1.plot(T, cero + 15, '--', c='k', alpha=0.5)
    w2.plot(T, cero, '--', c='k', alpha=0.5)
    r2.plot(T, cero, '--', c='k', alpha=0.5)
    p2.plot(T, cero, '--', c='k', alpha=0.5)
    q2.plot(T, cero, '--', c='k', alpha=0.5)
    plt.show()
    bar1.finish()
        
    
if len(sys.argv) == 1:
    hidden_sizes = [64,64,64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]


env = QuadcopterEnv()
env = NormalizedEnv(env)   
agent = DDPGagent(env,True,hidden_sizes)
noise = OUNoise(env.action_space)

un_grado = np.pi/180
env.lam = 0.0
print('d1 = ',env.d1)
print('d2 = ',env.d2)
print('lambda = ',env.lam)
print('arquitectura = ' + str(hidden_sizes))
env.p[9:] = np.ones(3)
train(agent, 1, 0.0, env, noise, 4)
agent.memory.remove()

data = pd.DataFrame(columns=('u', 'v', 'w', 'p','q','r','psi','theta','phi','x','y','z','score','tiempo','reward','porcentaje'))
data.u = vector_u; data.v = vector_v; data.w = vector_w
data.p = vector_p; data.q = vector_q; data.r = vector_r
data.psi = vector_psi; data.theta = vector_theta; data.phi = vector_phi
data.x = vector_x; data.y = vector_y; data.z = vector_z
data.score = vector_score; data.tiempo = vector_tiempo; data.reward = vector_reward ;data.porcentaje = vector_porcentaje

data.to_csv('PRUEBA1.csv',index = False)


        


