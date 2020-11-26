import torch
import numpy as np
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from .models_merge import Actor, Critic, weights_init
# from .models import Critic, weights_init
from .utils import Memory

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class DDPGagent:
    def __init__(self, env, flag=True, hidden_sizes=[64, 64], actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.98, tau=0.125, max_memory_size=int(1e7)):
        # Params
        self.num_states = 0
        if flag:
            self.num_states = 6
        self.num_states += env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

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
        return action

    def update(self, batch_size):
        # actions -> lambdas
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
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