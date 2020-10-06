import pathlib
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch
from datetime import datetime as dt
from DDPG.env.quadcopter_env import QuadcopterEnv, G, M, K, omega_0, STEPS, ZE, XE, YE, funcion
from Linear.step import control_feedback
from time import time 
from DDPG.utils import NormalizedEnv, OUNoise
from DDPG.ddpg import DDPGagent
from time import time
from numpy.linalg import norm
from tqdm import tqdm
from DDPG.load_save import load_nets, save_nets, remove_nets, save_buffer, remove_buffer
from tools.tools import imagen2d, reset_time, get_score, imagen_action
from tools.my_time import my_date
from numpy import pi, cos, sin
from numpy import remainder as rem


plt.style.use('ggplot')


now = my_date()
print('empezó:', now)
day = str(now['month']) +'_'+ str(now['day']) +'_'+ str(now['hr']) + str(now['min']) 
pathlib.Path('results/'+ day).mkdir(parents=True, exist_ok=True) 


today = str(dt.now())[5:10]
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


#u, v, w, p, q, r, psi, theta, phi, x, y, z = env.state
def training_loop(agent, env, noise):
    state = funcion(env.reset())
    noise.reset()
    episode_reward = 0
    score1 = 0
    score2 = 0
    s = 1
    while True:
        action = agent.get_action(state) # falta hacer el proceso inverso de normalización
        # action_
        action = noise.get_action(action, env.time[env.i])
        control = W0 + action 
        new_state, reward, done = env.step(control)
        s1, s2 = get_score(env.state, env)
        score1 += s1; score2 += s2
        episode_reward += reward
        agent.memory.push(state, action, reward, new_state, done)
        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)
        if done:
            break
        state = new_state
        s += 1
    return score1, score2, s, episode_reward


def train(agent, env, noise, episodes, k=0):
    E = []; R = []; S1 = []; S2 = []
    train_time = 0.0
    fig, axs = plt.subplots(3, 1)
    for episode in range(episodes):
        start_time = time()
        score1, score2, s, train_reward = training_loop(agent, env, noise)
        score1 /= s; score2 /= s
        train_time +=  time() - start_time
        E.append(episode); R.append(train_reward/s); S1.append(score1); S2.append(score2)
        vector_u.append(env.state[0]); vector_v.append(env.state[1]); vector_w.append(env.state[2])
        vector_p.append(env.state[6]); vector_q.append(env.state[7]); vector_r.append(env.state[8])
        vector_psi.append(env.state[9]); vector_theta.append(env.state[10]); vector_phi.append(env.state[11])
        vector_x.append(env.state[3]); vector_y.append(env.state[4]); vector_z.append(env.state[5])
        vector_score.append(score1)
        vector_tiempo.append(train_time)
        vector_reward.append(train_reward)
        vector_porcentaje.append(s/(env.tam-1) * 100)
    
    axs[0].plot(E, R, label='$p_'+str(k)+'$', c='c')
    axs[1].plot(E, S1, label='$p_'+str(k)+'$', c='m')
    axs[2].plot(E, S2, label='$p_'+str(k)+'$', c='y')

    axs[0].set_xlabel("Episodios", fontsize=21); 
    axs[1].set_xlabel("Episodios", fontsize=21) 
    axs[2].set_xlabel("Episodios", fontsize=21)
    axs[0].set_ylabel("Reward promedio", fontsize=21); 
    axs[1].set_ylabel("# objetivos promedio", fontsize=21) 
    axs[2].set_ylabel("# de veces entre (0, 22)", fontsize=21)
    axs[0].legend(loc=1)
    axs[1].legend(loc=1)
    axs[2].legend(loc=1)

    fig.set_size_inches(33.,21.)
    plt.savefig('results/'+day+'/score_reward_p'+str(k)+'.png', dpi=300)


def nsim(flag, n, show=True, k=0):
    fig, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
    alpha = 0.2
    total = 0
    for _ in range(n):
        t = env.time
        state = funcion(env.reset()); noise.reset()
        _, _, w, _, _, z, p, q, r, psi, theta, phi = env.state
        episode_reward = 0
        Z, W, Psi, R, Phi, P, Theta, Q, T = [], [], [], [], [], [], [], [], []
        X,Y = [], []
        env.flag  = flag
        while True:
            action = agent.get_action(state)
            real_action = env._action(action)
            # action = noise.get_action(action, env.time[env.i])
            control = W0 + real_action
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
                _, score = get_score(state, env)
                total += score
                break
        T = t[0:len(Z)]
        cero = np.zeros(len(Z))
        w1.plot(T, Z, c='b',alpha = alpha)
        w1.set_ylabel('z', fontsize=21)
        w2.plot(T, W, c='b',alpha = alpha)
        w2.set_ylabel('dz', fontsize=21) 
        r1.plot(T, Psi, c='r',alpha = alpha)
        r1.set_ylabel('$\psi$', fontsize=21)
        r2.plot(T, R, c='r',alpha = alpha)
        r2.set_ylabel('d$\psi$', fontsize=21)
        p1.plot(T, Phi, c='g',alpha = alpha)
        p1.set_ylabel('$\phi$', fontsize=21)
        p2.plot(T, P, c='g',alpha = alpha)
        p2.set_ylabel(' d$\phi$', fontsize=21)
        q1.plot(T, Theta,c = 'k',alpha = alpha)
        q1.set_ylabel('$ \\theta$', fontsize=21)
        q2.plot(T, Q,c = 'k' ,alpha = alpha)
        q2.set_ylabel(' d$ \\theta$', fontsize=21)
    w1.plot(T, cero + 15, '--', c='k', alpha=0.5)
    w2.plot(T, cero, '--', c='k', alpha=0.5)
    r2.plot(T, cero, '--', c='k', alpha=0.5)
    p2.plot(T, cero, '--', c='k', alpha=0.5)
    q2.plot(T, cero, '--', c='k', alpha=0.5)

    fig.suptitle("Vuelos terminados f="+str(total/n), fontsize=24)
    if show:
        plt.show()
    else:
        fig.set_size_inches(33.,21.)
        plt.savefig('results/'+day+'/nsim_p'+str(k)+'.png', dpi=300)
    return total/n 
        
    
if len(sys.argv) == 1:
    hidden_sizes = [64, 64, 64, 64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]


env = QuadcopterEnv()
env = NormalizedEnv(env)   
agent = DDPGagent(env,True,hidden_sizes)
noise = OUNoise(env.action_space)

un_grado = np.pi/180


p0 = np.zeros(12)
p1 = np.zeros(12); p1[3:6] = 0.5; p1[9:] = 0.5 * un_grado
p2 = np.zeros(12); p2[3:6] = 1; p2[9:] = 1 * un_grado
p3 = np.zeros(12); p3[3:6] = 1.5; p3[9:] = 1.5 * un_grado
p4 = np.zeros(12); p4[3:6] = 2; p4[9:] = 2 * un_grado
p5 = np.zeros(12); p5[3:6] = 2.5; p5[9:] = 2.5 * un_grado
p6 = np.zeros(12); p6[3:6] = 3; p6[9:] = 3 * un_grado
p7 = np.zeros(12); p7[3:6] = 3.5; p7[9:] = 3.5 * un_grado
p8 = np.zeros(12); p8[3:6] = 4; p8[9:] = 4 * un_grado
p9 = np.zeros(12); p9[3:6] = 4.5; p9[9:] = 4.5 * un_grado
p10 = np.zeros(12); p10[3:6] = 5; p10[9:] = 5 * un_grado


P = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
E = [500, 550, 600, 700, 800, 1000, 1200, 1600, 2000, 2800, 3600]

i = 0
n = 20 # simulaciones
old_frec = 0
subpath1 = day + "/best"
subpath2 = day +  "/recent"
for p, e in zip(P, E):
    env.p = p
    env.flag = False
    train(agent, env, noise, int(e), k=i)
    reset_time(env, 96000, 3600)
    frec = nsim(True, n, show=False, k=i)
    if frec > old_frec: # salvo la mejor red global
        remove_nets(subpath1); remove_buffer(subpath1)
        buffer = agent.memory.buffer
        save_nets(agent, hidden_sizes, subpath1); save_buffer(buffer, subpath1)
        old_frec = frec
    if frec >= 0.5: # salvo la red del paso anteror si termina la mitad de los vuelos
        remove_nets(subpath2); remove_buffer(subpath2)
        buffer = agent.memory.buffer
        save_nets(agent, hidden_sizes, subpath2); save_buffer(buffer, subpath2)

    reset_time(env, 800, 30)
    agent.memory.remove()
    i += 1


p13 = np.array([0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 5 * un_grado, 5 * un_grado])
env.p = p13
reset_time(env, 96000, 3600)
nsim(True, n, show=False, k=i)

p14 = np.array([0, 0, 0, 10, 10, 10, 0, 0, 0, 7 * un_grado, 7 * un_grado, 7 * un_grado])
env.p = p14
nsim(True, n, show=False, k=i+1)

P.append(p13)
P.append(p14)


data = pd.DataFrame(columns=('u', 'v', 'w', 'p','q','r','psi','theta','phi','x','y','z','score','tiempo','reward','porcentaje'))
data.u = vector_u; data.v = vector_v; data.w = vector_w
data.p = vector_p; data.q = vector_q; data.r = vector_r
data.psi = vector_psi; data.theta = vector_theta; data.phi = vector_phi
data.x = vector_x; data.y = vector_y; data.z = vector_z
data.score = vector_score; data.tiempo = vector_tiempo; data.reward = vector_reward ;data.porcentaje = vector_porcentaje

data.to_csv('results/'+day+'/final_state.csv', index=False)


data_p = pd.DataFrame(np.array(P), columns=['u', 'v', 'w', 'x', 'y', 'z', 'p', 'q', 'r', 'psi', 'theta', 'phi'])
data_p.to_csv('results/'+day+'/perturbations.csv', index=False)

print('terminó:', my_date())

        


