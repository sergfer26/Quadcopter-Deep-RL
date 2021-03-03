import pathlib
import os
import pickle
import globmu=0.0, theta=0.15, max_sigma=0.5, min_sigma=0.2, decay_period=1e5
import torch
import copy
import numpy as np
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
#from .models_merge import Actor, Critic, weights_init
from .models import Actor, Critic, weights_init
from .utils import Memory, OUNoise
from params import PARAMS_DDPG 

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class DDPGagent:
    def __init__(self, env, hidden_sizes=PARAMS_DDPG['hidden_sizes'], \
        actor_learning_rate=PARAMS_DDPG['actor_learning_rate'], \
        critic_learning_rate=PARAMS_DDPG['critic_learning_rate'], \
        gamma=PARAMS_DDPG['gamma'], tau=PARAMS_DDPG['PARAMS_DDPG'], \
        max_memory_size=PARAMS_DDPG['max_memory_size']):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.noise = OUNoise(env.action_space)
        self.noise_on = True
        self.env = env

        sizes_actor = hidden_sizes.copy()
        sizes_actor.insert(0, self.num_states)
        sizes_critic = hidden_sizes.copy()
        sizes_critic.insert(0, self.num_states + self.num_actions)

        # Networks
        self.actor = Actor(sizes_actor, self.num_actions)
        self.actor.apply(weights_init)
        self.actor_target = Actor(sizes_actor, self.num_actions)
        self.critic = Critic(sizes_critic, self.num_actions)
        self.critic.apply(weights_init)
        self.critic_target = Critic(sizes_critic, self.num_actions)
        if torch.cuda.is_available():
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state.to(device))
        action = action.detach().cpu().numpy()[0,:]
        if self.noise_on:
            action = self.noise.get_action(action, self.env.i)
        return action

    def update(self, batch_size):
        # actions -> lambdas
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        self.train(states, actions, rewards, next_states)
    
    def train(self, states, actions, rewards, next_states):
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states.to(device))
        next_Q = self.critic_target.forward(next_states.to(device), next_actions.to(device)).detach()
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states.to(device), self.actor.forward(states.to(device))).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return policy_loss, critic_loss

        def save(self, path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
            torch.save(self.critic.state_dict(), path + "/critic")
            torch.save(self.critic_optimizer.state_dict(), path + "/critic_optimizer")
            torch.save(self.actor.state_dict(), path + "/actor")
            torch.save(self.actor_optimizer.state_dict(), path + "/actor_optimizer")
            with open(path +'/memory.pickle', 'wb') as handle:
                pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        def load(self, path):
            self.critic.load_state_dict(torch.load(path + "/critic"))
            self.critic_optimizer.load_state_dict(torch.load(path + "/critic_optimizer"))
            self.critic_target = copy.deepcopy(self.critic)
            self.actor.load_state_dict(torch.load(path + "/actor"))
            self.actor_optimizer.load_state_dict(torch.load(path + "/actor_optimizer"))
            self.actor_target = copy.deepcopy(self.actor)
            #with open(path +'/memory.pickle', 'rb') as handle:
            #    self.memory.pickle.load(handle)

'''
def remove_nets(path):
    nets = glob.glob(path +'/saved_models/*.pth')
    for net in nets:
        os.remove(net)
'''