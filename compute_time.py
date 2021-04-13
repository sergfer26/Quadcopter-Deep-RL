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
#from progress.bar import Bar
from datetime import datetime
from get_report import create_report_ddpg
from params import PARAMS_TRAIN_DDPG
from time import time 
from simulation import sim, nSim, plot_nSim2D, plot_nSim3D
from correo import send_correo
import matplotlib.pyplot as plt
BATCH_SIZE = PARAMS_TRAIN_DDPG['BATCH_SIZE']
EPISODES = 2
TAU = 2 * pi #No es el tau del agente
SHOW = PARAMS_TRAIN_DDPG['SHOW']
  
def compute_time(agent, env):
    env.reset()
    TIMES = []
    MEMORY = []
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        t0 = time()
        while True:
            action = agent.get_action(state)
            action, reward, new_state, done = env.step(action)
            episode_reward += reward
            agent.memory.push(state, action, reward, new_state, done)
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)
            u, v, w, x, y, z, p, q, r, psi, theta, phi = env.state
            a1, a2, a3, a4 = env.action(action)
            if done:
                break
            state = new_state
        TIMES.append(time() - t0)
        MEMORY.append(agent.memory.__len__())
    plt.plot(MEMORY,TIMES,'.b')
    plt.xlabel('Tamaño de la memoria')
    plt.ylabel('Segundos')
    if SHOW:
        plt.show()
    else:
        plt.savefig('time.png')
        plt.close()




if __name__ == "__main__":
    env = AgentEnv(QuadcopterEnv())
    agent = DDPGagent(env)
    compute_time(agent, env)
