import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from DDPG.env.quadcopter_env import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, control_feedback
from time import time 
from DDPG.utils import NormalizedEnv, OUNoise
from DDPG.ddpg import DDPGagent
from time import time
from numpy.linalg import norm
from tqdm import tqdm
from DDPG.load_save import load_nets, save_nets
from tools.tools import imagen2d,imagen
from numpy import pi
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


def get_score(state):
    z = state[-1]
    w = state[2]
    if ZE - 0.20 < z < ZE + 0.20 and abs(w) < 0.25:
        return 1
    else:
        return 0


def reset_time(env, tamaño, tiempo_max):
    env.time_max = tiempo_max
    env.tam = tamaño
    env.time = np.linspace(0, env.time_max, env.tam)


def training_loop(agent, env, noise, pbar, test=False):
    state = env.reset()
    if not test:
        noise.reset()
    episode_reward = 0
    score = 0
    s = 1
    while True:
        action = agent.get_action(state)
        if not test:
            action = noise.get_action(action, env.time[env.i])
        W1 = control_feedback(state[-1] - 15, state[2], F1) * c1  # control z
        control = action + W0 + env.lam*W1
        new_state, reward, done = env.step(control)
        score += get_score(new_state)
        episode_reward += reward
        if not test:
            agent.memory.push(state, action, reward, new_state, done)
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)
        _, _, w, p, q, _, _, theta, phi, _, _, z = new_state
        pbar.set_postfix(R='{:.2f}'.format(episode_reward), # u='{:.2f}'.format(u), v='{:.2f}'.format(v), 
            w='{:.2f}'.format(w), p='{:.2f}'.format(p), q='{:2f}'.format(q), # r='{:.2f}'.format(r), psi='{:.2f}'.format(rem(psi, TAU)), 
                theta='{:.2f}'.format(rem(theta, TAU)), phi='{:.2f}'.format(rem(phi, TAU)), # x='{:.2f}'.format(x), y='{:.2f}'.format(y), 
                    z='{:.2f}'.format(z), s='{:.4f}'.format(noise.max_sigma))
        # pbar.set_postfix(R='{:.2f}'.format(episode_reward), w='{:.2f}'.format(w), z='{:.2f}'.format(z))
        pbar.update(1)
        if done:
            break
        state = new_state
        s += 1
    return score / s


def train(agent, rz, rw, env, noise, episodes):
    train_time = 0.0
    env.p[2], env.p[-1] = rw, rz
    max_sigma = ((16.6 / noise.decay_period) * noise.min_sigma)/( 16.6 /noise.decay_period -1)
    sigmas = np.linspace(noise.sigma, max_sigma, episodes)
    for episode in range(episodes):
        start_time = time()
        noise.max_sigma = sigmas[episode]
        with tqdm(total = env.tam, position=0) as pbar_train:
            pbar_train.set_description(f'Ep {episode + 1}/'+str(episodes)) #+' - training')
            train_score = training_loop(agent, env, noise, pbar_train)
            train_time +=  time() - start_time
            #writer_train.add_scalar('episode vs score', train_score, episode)
        #with tqdm(total = STEPS, position=0) as pbar_test:
        #    pbar_test.set_description(f'Episode {episode + 1}/'+str(episodes)+' - test')
        #    pbar_test.set_postfix(reward='0.0', w='0.0', z='0.0')
        #    test_score = training_loop(agent, env, noise, pbar_test, test=True)
        #    writer_train.add_scalar('episode vs score', test_score, episode)


def Sim(flag, agent, env):
    t = env.time
    state = env.reset()
    noise.reset()
    _, _, w, p, q, r, psi, theta, phi, _, _, z = state
    episode_reward = 0
    Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
    X,Y = [], []
    env.flag  = flag
    while True:
        # oz = z
        # z = state[-1] + np.random.normal(0, .02)
        # state[2] = (z - oz) * (env.tam / env.time_max)
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        W1 = control_feedback(z - 15, w, F1) * c1  # control z
        control = action + W0 + env.lam*W1  
        new_state, reward, done = env.step(control) 
        _, _, w, p, q, r, psi, theta, phi, x, y, z = state
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
        state = env.reset()
        noise.reset()
        _, _, w, p, q, r, psi, theta, phi, _, _, z = state
        episode_reward = 0
        Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
        X,Y = [], []
        env.flag  = flag
        while True:
            action = agent.get_action(state)
            action = noise.get_action(action, env.time[env.i])
            control = action + W0 + env.lam*W1
            new_state, reward, done = env.step(control) 
            _, _, w, p, q, r, psi, theta, phi, x, y, z = state
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
agent = DDPGagent(env,hidden_sizes)
noise = OUNoise(env.action_space)
#writer_train = SummaryWriter()
#writer_test = SummaryWriter()

#load_nets(agent,hidden_sizes)

RA = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2,2.5,3,3.5,4,4.5,5]
un_grado = np.pi/180
env.d = 1
print('d = ',env.d)

env.p[6],env.p[7],env.p[8] = 0.1*un_grado*np.ones(3)
train(agent, 1, 0.0, env, noise, 200)
agent.memory.remove()
env.p[6],env.p[7],env.p[8] = 0.2*un_grado*np.ones(3)
train(agent, 1, 0.2, env, noise, 200)
#env.d = 0.5

#for ra in RA:
#    env.p[6],env.p[7],env.p[8] = ra*un_grado*np.ones(3)
#    train(agent, 1, 0.2, env, noise, 50)

#save_nets(agent, hidden_sizes)

'''
for rz, rw, e in zip(RZ, RW, E):
    train(agent, rz, rw, env, noise, e, writer_train, writer_test)
    save_nets(agent, hidden_sizes)
    Sim(True, agent, env)
    noise.max_sigma = 1.0
    noise.sigma = 1.0


noise.max_sigma = 0.0
noise.sigma = 0.0
reset_time(env, 800, 30)
Sim(True, agent, env)
'''

def sim():
    tem = env.p
    env.p *= 0
    Sim(True,agent,env)
    env.p = tem

def clear():
    for _ in range(100):
        print('>>>')
        
def render():
    t = env.time
    state = env.reset()
    noise.reset()
    _, _, _, _, _, _, psi, theta, phi, x, y, z = state
    X,Y,Z = [], [], []
    Psi, Phi,Theta = [0], [0], [0]
    i = 0
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        control = action + W0
        new_state, _, done = env.step(control)
        _, _, _, _, _, _, psi, theta, phi, x, y, z = state
        X.append(x);Y.append(y);Z.append(z)
        Psi.append(psi);Phi.append(phi);Theta.append(theta)
        state = new_state
        if done:
            break
    scene.width = scene.height =900
    scene.background = color.gray(0.7)
    L = 70
    scene.center = vec(0.05*L,0.2*L,0)
    scene.range = 1.3*L
    R = L/100
    d = L-2
    ball = box(pos=vector(0,15,0), size = vector(10,0.2,10), color=color.red)
    piso = box(pos=vector(0,0,0), size = vector(100,0.1,100), color=color.green)
    #floor = sphere(pos=vector(x,z,y), radius=0.5, color=color.red)
    xaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=0.5*R, color=color.yellow)
    yaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=0.5*R, color=color.yellow)
    zaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=0.5*R, color=color.yellow)
    k = 1.02
    h = 0.05*L
    text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h, align='center', billboard=True, emissive=True)
    text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h, align='center', billboard=True, emissive=True)
    text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h, align='center', billboard=True, emissive=True)
    c = curve(color = color.blue)
    while True:
        rate(50)
        if i < len(X):
            #label( pos=vec(5,20,5), text=str(i) )
            ball.pos.x , ball.pos.y,ball.pos.z = X[i],Z[i],Y[i]
            ball.rotate(angle = Theta[i+1] - Theta[i],axis = vector(1,0,0))
            ball.rotate(angle = Psi[i+1] - Psi[i],axis = vector(0,1,0))
            ball.rotate(angle = Theta[i+1] -Theta[i],axis = vector(0,0,1))
            c.append(ball.pos)
            i+=1
        else:
            label(pos=vec(5,20,5), text='La simulación terminó')

