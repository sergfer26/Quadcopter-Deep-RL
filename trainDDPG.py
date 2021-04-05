import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import pytz
import pandas as pd
import pathlib 
from tqdm import tqdm
from quadcopter_env import QuadcopterEnv, AgentEnv, omega_0, STEPS
from simulation import nSim3D, nSim, sim
from DDPG.ddpg import DDPGagent
from numpy.linalg import norm
from numpy import pi, cos, sin
from numpy import remainder as rem
#from progress.bar import Bar
from datetime import datetime
from get_report import create_report_ddpg
from params import PARAMS_TRAIN_DDPG
from simulation import sim, nSim, plot_nSim2D, plot_nSim3D


BATCH_SIZE = PARAMS_TRAIN_DDPG['BATCH_SIZE']
EPISODES = PARAMS_TRAIN_DDPG['EPISODES']
n = PARAMS_TRAIN_DDPG['n']
TAU = 2 * pi #No es el tau del agente
SHOW = PARAMS_TRAIN_DDPG['SHOW']
  

def train(agent, env):
    #s = 1
    R = []
    env.reset()
    for episode in range(EPISODES):
        with tqdm(total=env.steps, position=0) as pbar:
            pbar.set_description(f'Ep {episode + 1}/'+str(EPISODES))
            state = env.reset()
            episode_reward = 0
            while True:
                action = agent.get_action(state)
                action, reward, new_state, done = env.step(action)
                episode_reward += reward
                agent.memory.push(state, action, reward, new_state, done)
                if len(agent.memory) > BATCH_SIZE:
                    agent.update(BATCH_SIZE)
                u, v, w, x, y, z, p, q, r, psi, theta, phi = env.state
                a1, a2, a3, a4 = env.action(action)
                pbar.set_postfix(R='{:.2f}'.format(episode_reward),
                    w='{:.2f}'.format(w), v='{:.2f}'.format(v), u='{:.2f}'.format(u), 
                    p='{:.2f}'.format(p), q='{:2f}'.format(q), r='{:.2f}'.format(r),
                    psi='{:.2f}'.format(rem(psi, TAU)), theta='{:.2f}'.format(rem(theta, TAU)), phi='{:.2f}'.format(rem(phi, TAU)), 
                    z='{:.2f}'.format(z), y='{:.2f}'.format(y), x='{:.2f}'.format(x)) 
                pbar.update(1)
                if done:
                    break
                state = new_state
            R.append(episode_reward)
    return R


if __name__ == "__main__":
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute

    PATH = 'results_ddpg/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
    if not SHOW:
        pathlib.Path(PATH).mkdir(parents=True, exist_ok=True) 
    
    env = AgentEnv(QuadcopterEnv())
    #env.flag = False
    agent = DDPGagent(env)
    CR_t = train(agent, env)
    agent.noise_on = False
    agent.save(PATH)
    plt.plot(CR_t)
    plt.xlabel('episodes')
    plt.title(r'$r_t = \mathbb{1}_{x <= g + 1} - 0.01 \|x - g\| - 0.01 \|[dx, d\theta]\| - 0.5 \|I - X_{\theta}\|$')
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/c_rewards.png', bbox_inches='tight')
        plt.close()
    
    n_states, n_actions, n_scores = nSim(False, agent, env, 10)
    columns = ('$u$', '$v$', '$w$', '$x$', '$y$', '$z$', '$p$', '$q$', '$r$', r'$\psi$', r'$\theta$', r'$\varphi$')
    plot_nSim2D(n_states, columns, env.time, show=SHOW, file_name=PATH + '/sim_states.png')
    columns = ['$a_{}$'.format(i) for i in range(1,5)] 
    plot_nSim2D(n_actions, columns, env.time, show=SHOW, file_name=PATH + '/sim_actions.png')
    columns = ('$r_t$', '$Cr_t$', 'is stable', 'cotained')
    plot_nSim2D(n_scores, columns, env.time, show=SHOW, file_name=PATH + '/sim_scores.png')
    plot_nSim3D(n_states, show=SHOW, file_name=PATH + '/sim_flights.png')
    if not SHOW:
        create_report_ddpg(PATH)
