import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, h_sizes):
        super(Critic, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        
        self.out = nn.Linear(h_sizes[-1], 1)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.out(x)

        return x

class Actor(nn.Module):
    def __init__(self, h_sizes, output_size):
        super(Actor, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        
        self.out = nn.Linear(h_sizes[-1], output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """

        x = state
        # import pdb; pdb.set_trace()
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = torch.tanh(self.out(x))
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)