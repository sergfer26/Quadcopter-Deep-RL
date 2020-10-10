import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from DDPG.env.quadcopter_env import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, funcion
from Linear.step import control_feedback, imagen_accion
from time import time 
from DDPG.utils import NormalizedEnv, OUNoise
from DDPG.ddpg import DDPGagent
from time import time
from numpy.linalg import norm
from tqdm import tqdm
from DDPG.load_save import load_nets, save_nets
from tools.tools import imagen2d, imagen, reset_time, get_score, imagen_action
from numpy import pi, cos, sin
from numpy import remainder as rem
from progress.bar import Bar, ChargingBar


TAU = 2 * pi
BATCH_SIZE = 32

W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0

def agent_action(agent, env, noise, state):
    action = agent.get_action(state)
    real_action = env._action(action)
    real_action = noise.get_action(real_action, env.time[env.i])
    control = real_action + W0
    new_state, reward, done = env.step(control)
    return real_action, action, new_state, reward, done

    

def training_loop(agent, env, noise, pbar):
    state = funcion(env.reset())
    noise.reset()
    episode_reward = 0
    s = 1
    while True:
        _, action, new_state, reward, done = agent_action(agent, env, noise, state)
        episode_reward += reward
        agent.memory.push(state, action, reward, new_state, done)
        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)
        _, _, w, _, _, z, p, q, _, _, theta, phi = env.state
        pbar.set_postfix(R='{:.2f}'.format(episode_reward),
            w='{:.2f}'.format(w), p='{:.2f}'.format(p), q='{:2f}'.format(q),
                theta='{:.2f}'.format(rem(theta, TAU)), phi='{:.2f}'.format(rem(phi, TAU)), 
                    z='{:.2f}'.format(z), s='{:.4f}'.format(noise.max_sigma))
        pbar.update(1)
        if done:
            break
        state = new_state
        s += 1
    return episode_reward/s


def train(agent, env, noise, episodes): #, axs=None):
    train_time = 0.0
    for episode in range(episodes):
        start_time = time()
        with tqdm(total = env.tam, position=0) as pbar_train:
            pbar_train.set_description(f'Ep {episode + 1}/'+str(episodes)) 
            reward = training_loop(agent, env, noise, pbar_train)
            train_time +=  time() - start_time
    print(train_time)


def Sim(flag, agent, env):
    t = env.time
    state = funcion(env.reset())
    noise.reset()
    _, _, w, _, _, z, p, q, r, psi, theta, phi  = env.state
    episode_reward = 0
    Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
    X, Y = [], []
    acciones = []
    env.flag  = flag
    while True:
        real_action, action, new_state, reward, done = agent_action(agent, env, noise, state)
        _, _, w, x, y, z, p, q, r, psi, theta, phi = env.state
        Z.append(z); W.append(w)
        Psi.append(psi); R.append(r)
        Phi.append(phi); P.append(p)
        Theta.append(theta); Q.append(q)
        X.append(x); Y.append(y)
        acciones.append(real_action)
        state = new_state
        episode_reward += reward
        if done:
            break
    T = t[0:len(Z)]
    imagen2d(Z, W, Psi, R, Phi, P, Theta, Q, T)
    imagen_accion(acciones, T)


def nsim(flag, n):
    fig, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
    bar1 = Bar('Procesando:', max=n)
    alpha = 0.2
    for _ in range(n):
        bar1.next()
        t = env.time
        state = funcion(env.reset())
        noise.reset()
        _, _, w, _, _, z, p, q, r, psi, theta, phi  = env.state
        episode_reward = 0
        Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
        X,Y = [], []
        env.flag  = flag
        while True:
            _, _, new_state, reward, done = agent_action(agent, env, noise, state)
            _, _, w, x, y, z, p, q, r, psi, theta, phi = env.state
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


def sim():
    tem = env.p
    env.p *= 0
    Sim(True,agent,env)
    env.p = tem


def clear():
    for _ in range(100):
        print(' ')


if len(sys.argv) == 1:
    hidden_sizes = [64, 64, 64, 64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]


env = QuadcopterEnv()
env = NormalizedEnv(env)   
agent = DDPGagent(env, hidden_sizes=hidden_sizes)
noise = OUNoise(env.action_space)

#load_nets(agent,hidden_sizes)

un_grado = np.pi/180
env.d = 1
E = 2 * np.ones(1)

p0 = np.zeros(12)
p1 = np.zeros(12); p1[-1] = 1 * un_grado
p2 = np.zeros(12); p2[-2] = 1 * un_grado
p3 = np.zeros(12); p3[5] = 2; p3[2] = 0.5
p4 = np.zeros(12); p4[-1] = 2 * un_grado; p4[-6] = 0.1
p5 = np.zeros(12); p4[-5] = 2 * un_grado; p4[-5] = 0.1
p6 = np.array([0, 0, 0.6, 0, 0, 1.5, 0.2, 0.2, 0, 0, 1.5 * un_grado, 1.5 * un_grado])
P = [p0] # , p1, p2, p3, p4, p5, p6]

env.flag = False

for p, e in zip(P, E):
    env.p = p
    train(agent, env, noise, int(e))
    reset_time(env, 96000, 3600)
    nsim(True, 20)
    env.flag = False
    reset_time(env, 800, 30)
    agent.memory.remove()


reset_time(env, 96000, 3600)
Sim(True, agent, env)
reset_time(env, 800, 30)






