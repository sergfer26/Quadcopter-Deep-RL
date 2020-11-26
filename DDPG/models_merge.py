import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable


class Actor(nn.Module):
    def __init__(self, h_sizes, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.dropout = False
        self.p = 0.5
        h_sizes[0] = 9
        size1, self.hidden1 = self.get_hidden_layers(h_sizes)
        size2, self.hidden2 = self.get_hidden_layers(h_sizes[0:-1])

        self.mid = nn.Linear(size1 + size2, h_sizes[-1])
        self.out = nn.Linear(h_sizes[-1], output_size)

    def get_hidden_layers(self, h_sizes):
        hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        
        return h_sizes[k+1], hidden
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """

        x1 = state[:, 0:9]; x2 = state[:, 9:]
        for layer in self.hidden1:
            x1 = F.relu(layer(x1))
            if self.dropout:
                x1 = F.dropout(x1, p=self.p, training=self.training)
        
        for layer in self.hidden2:
            x2 = layer(x2)
            if self.droput:
                x2 = F.dropout(x2, p=self.p, training=self.training)

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.mid(x))
        x = torch.tanh(self.out(x))

        return x


class Critic(nn.Module):

    def __init__(self, h_sizes, output_size):
        super(Critic, self).__init__()
        h_sizes[0] = 9
        size1, self.hidden1 = self.get_hidden_layers(h_sizes)
        size2, self.hidden2 = self.get_hidden_layers(h_sizes[0:-1])

        h_sizes[0] = 4
        size3, self.hidden3 = self.get_hidden_layers(h_sizes[0:-1])
    
        self.mid = nn.Linear(size1 + size2 + size3, h_sizes[-1])
        self.out = nn.Linear(h_sizes[-1], output_size)

    def get_hidden_layers(self, h_sizes):
        hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        
        return h_sizes[k+1], hidden

    def forward(self, state, action):


        x1 = state[:, 0:9]; x2 = state[:, 9:]; a = action
        for layer in self.hidden1:
            x1 = F.relu(layer(x1))
        
        for layer in self.hidden2:
            x2 = layer(x2)

        for layer in self.hidden3:
            a = layer(a)

        x = torch.cat((x1, x2, a), dim=1)

        x = F.relu(self.mid(x))
        x = self.out(x)

        return x

    


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)