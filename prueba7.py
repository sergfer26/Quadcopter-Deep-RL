import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from DDPG.env.quadcopter_env_new import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, funcion
from Linear.step import get_control, control_feedback
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
    

def training_loop(agent, env, noise, pbar):
    state = funcion(env.reset())
    noise.reset()
    episode_reward = 0
    score = 0
    s = 1
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        # W = get_control(state[0:12], ZE)
        #control = W0.T[0] + env.lam * action + (1 - env.lam) * W.T[0]
        control = action + W0
        new_state, reward, done = env.step(control)
        score += get_score(new_state, env)
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
    return score/s, episode_reward/s


def train(agent, env, noise, episodes): #, axs=None):
    E = []; R = []; S = []
    train_time = 0.0
    #max_sigma = ((16.6 / noise.decay_period) * noise.min_sigma)/( 16.6 /noise.decay_period -1)
    #sigmas = np.linspace(noise.sigma, max_sigma, episodes)
    for episode in range(episodes):
        start_time = time()
        #noise.max_sigma = max(0, sigmas[episode])
        with tqdm(total = env.tam, position=0) as pbar_train:
            pbar_train.set_description(f'Ep {episode + 1}/'+str(episodes)) 
            score, reward = training_loop(agent, env, noise, pbar_train)
            train_time +=  time() - start_time
            E.append(episode); R.append(reward); S.append(score)
    print(train_time)

    #if axs.all():
    #    axs[0].plot(E, R, label='$\lambda=$'+str(env.lam))
    #    axs[1].plot(E, S, label='$\lambda=$'+str(env.lam))


def Sim(flag, agent, env):
    t = env.time
    state = funcion(env.reset())
    noise.reset()
    _, _, w, _, _, z, p, q, r, psi, theta, phi  = env.state
    episode_reward = 0
    Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
    X, Y = [], []
    env.flag  = flag
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        #control_ = get_control(state, ZE)
        #control = W0.T[0] + env.lam * action + (1 - env.lam) * control_.T[0]
        control = action + W0
        new_state, reward, done = env.step(control) 
        _, _, w, x, y, z, p, q, r, psi, theta, phi = env.state
        Z.append(z); W.append(w)
        Psi.append(psi); R.append(r)
        Phi.append(phi); P.append(p)
        Theta.append(theta); Q.append(q)
        X.append(x); Y.append(y)
        state = new_state
        episode_reward += reward
        if done:
            break
    T = t[0:len(Z)]
    imagen2d(Z, W, Psi, R, Phi, P, Theta, Q, T)
    # imagen_action([A[:, 0], A[:, 1], A[:, 2], A[:, 3]], T)
    imagen(X, Y, Z)


def nsim(flag, n):
    fig, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots((4, 2), sharex=True, figsize=(50,100))
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
            action = agent.get_action(state)
            action = noise.get_action(action, env.time[env.i])
            #control_ = get_control(state, ZE)
            #control = W0.T[0] + env.lam * action + (1 - env.lam) * control_.T[0]
            control = action + W0
            new_state, reward, done = env.step(control) 
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
    hidden_sizes = [64, 256, 64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]


env = QuadcopterEnv()
env = NormalizedEnv(env)   
agent = DDPGagent(env, hidden_sizes)
noise = OUNoise(env.action_space)
#writer_train = SummaryWriter()
#writer_test = SummaryWriter()

#load_nets(agent,hidden_sizes)

un_grado = np.pi/180
env.d = 1
env.lam = 1

# du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi

fig, axs = plt.subplots((2, 1), sharex=True, figsize=(100, 50))

# lambdas = [.05, .1, .4, .7, .8, .85, .9, .95, .97, .98, .99, 1]
E = 500 * np.ones(7)

p0 = np.zeros(12)
p1 = np.zeros(12); p1[-1] = 1 * un_grado
p2 = np.zeros(12); p2[-2] = 1 * un_grado
p3 = np.zeros(12); p3[5] = 2; p3[2] = 0.5
p4 = np.zeros(12); p4[-1] = 2 * un_grado; p4[-6] = 0.1
p5 = np.zeros(12); p4[-5] = 2 * un_grado; p4[-5] = 0.1
p6 = np.array([0, 0, 0.6, 0, 0, 1.5, 0.2, 0.2, 0, 0, 1.5 * un_grado, 1.5 * un_grado])
P = [p0, p1, p2, p3, p4, p5, p6]

env.flag = False

for p, e in zip(P, E):
    env.p = p
    train(agent, env, noise, int(e))
    reset_time(env, 96000, 3600)
    nsim(True, 20)
    reset_time(env, 800, 30)
    agent.memory.remove()


reset_time(env, 96000, 3600)
Sim(True, agent, env)
reset_time(env, 800, 30)


axs[0].set_xlabel("Episodios"); axs[1].set_xlabel("Episodios")
axs[0].set_ylabel("Reward promedio"); axs[1].set_ylabel("# objetivos promedio")
axs[0].legend(loc=1)
axs[1].legend(loc=1)
plt.show()




