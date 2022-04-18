import torch
import copy
import numpy as np
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from .cost import CostNN


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

REWARD_FUNCTION_UPDATE = 5
DEMO_SIZE = 64
SAMP_SIZE = 256


class GCL:

    def __init__(self, expert, agent, env):
        self.expert = expert
        self.agent = agent
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.cost = CostNN(state_dim, action_dim)
        self.cost_optimizer = torch.optim.Adam(
            self.cost.parameters(), 1e-2, weight_decay=1e-4)
        self.memory_demo = expert.memory_traj
        self.memory_samp = agent.memory_traj

    def train_agent(self):
        pass

    def distribution_traj(self, traj):
        states, actions, next_states = traj
        return 1

    def train_cost(self):
        for _ in range(REWARD_FUNCTION_UPDATE):
            data_demo = self.memory_demo.sample(DEMO_SIZE)
            data_samp = self.memory_samp.sample(SAMP_SIZE)
            data_samp.extend(data_demo)
            data_samp = np.array(data_samp)
            data_demo = np.array(data_demo)
            states, actions, next_states = data_samp[:, 0], data_samp[:, 1], data_samp[:, 2]
            states_demo, actions_demo = data_demo[:, 0], data_demo[:, 1]
            
            # Reducing from float64 to float32 for making computaton faster
            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
            states_demo_tensor = torch.tensor(states_demo, dtype=torch.float32)
            actions_demo_tensor = torch.tensor(actions_demo, dtype=torch.float32)

            cost_samp = self.cost(torch.cat((states_tensor, actions_tensor), dim=-1))
            cost_samp = self.cost(torch.cat((states_tensor, actions_tensor), dim=-1))
            

