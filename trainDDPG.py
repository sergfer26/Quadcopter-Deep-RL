import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import pytz
import pandas as pd
import pathlib 
from tqdm import tqdm
from quadcopter_env import QuadcopterEnv, AgentEnv, G, M, K, omega_0, STEPS, ZE, XE, YE
#from Linear.step import control_feedback, F1, F2, F3, F4, c1, c2, c3, c4
from DDPG.ddpg import DDPGagent
from numpy.linalg import norm
#from DDPG.load_save import load_nets, save_nets, remove_nets, save_buffer, remove_buffer
#from tools.tools import imagen2d, imagen_action, sub_plot_state
from numpy import pi, cos, sin
from numpy import remainder as rem
from progress.bar import Bar
from datetime import datetime
from get_report import create_report
from params import PARAMS_TRAIN
from graphics import*

BATCH_SIZE = PARAMS_TRAIN['BATCH_SIZE']
EPISODES = PARAMS_TRAIN['EPISODES']
TAU = 2 * pi #No es el tau del agente
SHOW = PARAMS_TRAIN['SHOW']
  
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
                    #w='{:.2f}'.format(w), v='{:.2f}'.format(v), u='{:.2f}'.format(u), 
                    #p='{:.2f}'.format(p), q='{:2f}'.format(q), r='{:.2f}'.format(r),
                    a1='{:.2f}'.format(a1), a2='{:.2f}'.format(a2), a3='{:.2f}'.format(a3), a4='{:.2f}'.format(a4),
                    psi='{:.2f}'.format(rem(psi, TAU)), theta='{:.2f}'.format(rem(theta, TAU)), phi='{:.2f}'.format(rem(phi, TAU)), 
                    z='{:.2f}'.format(z), y='{:.2f}'.format(y), x='{:.2f}'.format(x)) 
                pbar.update(1)
                if done:
                    break
                state = new_state
            R.append(episode_reward)
    return R


def sim(flag, agent, env):
    #t = env.time
    env.flag  = flag
    state = env.reset()
    states = np.zeros((int(env.steps + 1), env.observation_space.shape[0] - 6))
    actions = np.zeros((int(env.steps), env.action_space.shape[0]))
    scores = np.zeros((int(env.steps), 4)) # r_t, Cr_t, stable, contained
    states[0, :]= env.reverse_observation(state)
    episode_reward = 0
    i = 0
    while True:
        action = agent.get_action(state)
        action, reward, new_state, done = env.step(action)
        episode_reward += reward
        states[i + 1, :] = env.state
        actions[i, :] = env.action(action)
        scores[i, :] = np.array([reward, episode_reward, env.is_stable(new_state), env.is_contained(new_state)])
        state = new_state
        if done:
            break
        i += 1
    return states, actions, scores


def nSim(flag, agent, env, n):
    states = np.zeros((env.time_max + 1, env.observation_space.shape[0] - 6, n))
    actions = np.zeros((env.time_max, env.action_space[0], n))
    scores = np.zeros((env.time_max, 4, n))
    env.flag  = flag
    bar = Bar('Processing', max=n)
    for k in range(n):
        bar.next()
        state = env.reset()
        states[0, :, k]  = env.state
        episode_reward = 0
        i = 0
        while True:
            action = agent.get_action(state)
            action, reward, new_state, done = env.step(action)
            episode_reward += reward
            states[i + 1, :, k] = env.state
            actions[i, :, k] = env.action(action)
            scores[i, :, k] = np.array([reward, episode_reward, env.is_stable(), env.is_contained()])
            state = new_state
            if done:
                break
            i += 1
    return states, actions, scores


if __name__ == "__main__":
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    month = mexico_now.month
    day = mexico_now.day
    hour = mexico_now.hour
    minute = mexico_now.minute

    PATH = 'results_ddpg/'+ str(month) + '_'+ str(day) +'_'+ str(hour) + str(minute)
    pathlib.Path(PATH).mkdir(parents=True, exist_ok=True) 
    
    env = AgentEnv(QuadcopterEnv())
    #env.flag = False
    agent = DDPGagent(env)
    CR_t = train(agent, env)
    agent.noise_on = False
    agent.save(PATH)
    states, actions, scores = sim(True, agent, env)
    plt.plot(CR_t)
    plt.xlabel('episodes')
    plt.title(r'$r_t = \mathbb{1}_{x <= g + 1} - 0.01 \|x - g\| - 0.01 \|[dx, d\theta]\| - 0.5 \|I - X_{\theta}\|$')
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/c_rewards.png')
        plt.close()
    columns=('$u$', '$v$', '$w$', '$x$', '$y$', '$z$', '$p$', '$q$', '$r$', '$\psi$', r'$\theta$', '$\phi$')
    statesDF = pd.DataFrame(states, columns=columns)
    statesDF.plot(subplots=True, layout=(4, 3), figsize=(10, 7))
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_states.png')
        plt.close()
    columns=('$a_1$', '$a_2$', '$a_3$', '$a_4$')
    actionsDF = pd.DataFrame(actions, columns=columns)
    actionsDF.plot(subplots=True, layout=(2, 2), figsize=(10, 7))
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_actions.png')
        plt.close()
    columns=('$r_t$', '$Cr_t$', 'Stable', 'Contained')
    scoresDF = pd.DataFrame(scores, columns=columns)
    scoresDF.plot(subplots=True, layout=(2, 2), figsize=(10, 7))
    if SHOW:
        plt.show()
    else:
        plt.savefig(PATH + '/sim_scores.png')
        plt.close()

    nsim3D(10, agent, env, PATH)
    create_report(PATH)