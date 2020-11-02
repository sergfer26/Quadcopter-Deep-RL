import pathlib
import pandas as pd
import numpy as np
from trainRL import *
from time import time
from tools.my_time import my_date
from datetime import datetime as dt

plt.style.use('ggplot')

now = my_date()
print('empezó:', now)
DAY = str(now['month']) +'_'+ str(now['day']) +'_'+ str(now['hr']) + str(now['min'])
PATH = 'resultsRL/'+ DAY
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)


vector_u, vector_v, vector_w = [],[],[]
vector_p, vector_q, vector_r = [],[],[]
vector_psi, vector_theta, vector_phi = [],[],[]
vector_x, vector_y, vector_z = [],[],[]
vector_score,vector_tiempo,vector_reward,vector_porcentaje = [],[],[],[]


if len(sys.argv) == 1:
    hidden_sizes = [64, 64, 64]
else:
    hidden_sizes = sys.argv[1:]
    hidden_sizes = [int(i) for i in hidden_sizes]


env = QuadcopterEnv()
env = NormalizedEnv(env)   
agent = DDPGagent(env, hidden_sizes=hidden_sizes)
noise = OUNoise(env.action_space)
un_grado = np.pi/180
env.d = 1


def sim(flag):
    path = PATH + '/sims/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    Sim(flag, agent, env, noise, show=False, path=path)


def nsim(flag, n, k=0):
    path = PATH + '/nsims'
    file_path = path +'/nsim_p'+ str(k) +'.png'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    mean_reward = nSim(flag, agent, env, noise, n, show=False, path=file_path)
    return mean_reward


def train(agent, env, noise, episodes, path=PATH, k=0):
    E = []; R = []; S1 = []; S2 = []
    train_time = 0.0
    fig, axs = plt.subplots(3, 1)
    for episode in range(episodes):
        score1, score2, train_reward = training_loop(agent, env, noise)
        E.append(episode); R.append(train_reward); S1.append(score1); S2.append(score2)
        vector_u.append(env.state[0]); vector_v.append(env.state[1]); vector_w.append(env.state[2])
        vector_p.append(env.state[6]); vector_q.append(env.state[7]); vector_r.append(env.state[8])
        vector_psi.append(env.state[9]); vector_theta.append(env.state[10]); vector_phi.append(env.state[11])
        vector_x.append(env.state[3]); vector_y.append(env.state[4]); vector_z.append(env.state[5])
        vector_score.append(score1)
        vector_tiempo.append(train_time)
        vector_reward.append(train_reward)
        vector_porcentaje.append(800/(env.tam-1) * 100)

    axs[0].plot(E, R, label='$p_'+str(k)+'$', c='c')
    axs[1].plot(E, S1, label='$p_'+str(k)+'$', c='m')
    axs[2].plot(E, S2, label='$p_'+str(k)+'$', c='y')

    axs[0].set_xlabel("Episodios", fontsize=21); 
    axs[1].set_xlabel("Episodios", fontsize=21) 
    axs[2].set_xlabel("Episodios", fontsize=21)
    axs[0].set_ylabel("Reward promedio", fontsize=21); 
    axs[1].set_ylabel("# objetivos promedio", fontsize=21) 
    axs[2].set_ylabel("# de veces dentro del cuarto", fontsize=21)
    axs[0].legend(loc=1)
    axs[1].legend(loc=1)
    axs[2].legend(loc=1)

    fig.set_size_inches(33.,21.)
    path = path +'/nsims'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path +'/score_reward_p'+str(k)+'.png', dpi=300)


p0 = np.zeros(12)
p1 = np.zeros(12); p1[3:6] = 0.5
p2 = np.zeros(12); p2[3:6] = 1
p3 = np.zeros(12); p3[3:6] = 1.5
p4 = np.zeros(12); p4[3:6] = 2
p5 = np.zeros(12); p5[3:6] = 2.5
p6 = np.zeros(12); p6[3:6] = 3
p7 = np.zeros(12); p7[3:6] = 3.5
p8 = np.zeros(12); p8[3:6] = 4
p9 = np.zeros(12); p9[3:6] = 4.5
p10 = np.zeros(12); p10[3:6] = 5
p11 = np.zeros(12); p11[3:6] = 5; p11[9:] = 0.5 * un_grado
p12 = np.zeros(12); p12[3:6] = 5; p12[9:] = 1 * un_grado
p13 = np.zeros(12); p13[3:6] = 6; p13[9:] = 1 * un_grado


P = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]
E = [250 , 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]



i = 0
n = 10 # simulaciones
old_frec = 0
subpath1 = PATH + "/models/best"; subpath1_ = PATH + "/buffer/best"
subpath2 = PATH +  "/models/recent"; subpath2_ = PATH +  "/buffer/recent"
best_reward = - np.inf
for p, e in zip(P, E):
    env.p = p
    env.flag = False
    train(agent, env, noise, e, k=i)
    env.set_time(96000, 3600)
    mean_reward = nsim(True, 10, k=i)
    if best_reward <= mean_reward:
        best_reward = mean_reward
        buffer = agent.memory.buffer
        save_nets(agent, hidden_sizes, subpath1)
        save_buffer(buffer, subpath1_)
    env.set_time(800, 30)
    agent.memory.remove()
    i += 1

buffer = agent.memory.buffer
save_nets(agent, hidden_sizes, subpath2)
save_buffer(buffer, subpath2_)


env.set_time(96000, 3600)
sim(True)
env.set_time(800, 30)

data = pd.DataFrame(columns=('u', 'v', 'w', 'p','q','r','psi','theta','phi','x','y','z','score','tiempo','reward','porcentaje'))
data.u = vector_u; data.v = vector_v; data.w = vector_w
data.p = vector_p; data.q = vector_q; data.r = vector_r
data.psi = vector_psi; data.theta = vector_theta; data.phi = vector_phi
data.x = vector_x; data.y = vector_y; data.z = vector_z
data.score = vector_score; data.tiempo = vector_tiempo; data.reward = vector_reward ;data.porcentaje = vector_porcentaje

data.to_csv('resultsRL/'+DAY+'/final_state.csv', index=False)


data_p = pd.DataFrame(np.array(P), columns=['u', 'v', 'w', 'x', 'y', 'z', 'p', 'q', 'r', 'psi', 'theta', 'phi'])
data_p.to_csv('resultsRL/'+DAY+'/perturbations.csv', index=False)

print('terminó:', my_date())

