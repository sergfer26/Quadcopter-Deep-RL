import torch
import numpy as np
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from torch.autograd import Variable
from scipy.stats import multivariate_normal

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class Policy(nn.Module):
    def __init__(self, env, hidden_sizes, is_stochastic=True):
        super(Policy, self).__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.env = env
        self.state_dim = state_dim
        self.is_stochastic = is_stochastic

        self._sigma = np.identity(action_dim)
        # Definición de la arquitectura
        if not isinstance(hidden_sizes, list):
            h_sizes = [64, 64]
        else:
            h_sizes = hidden_sizes.copy()
        h_sizes.insert(0, state_dim)
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        self.out = nn.Linear(h_sizes[-1], action_dim)

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

    def to_numpy(self, x, t_x=None):
        if callable(t_x):
            x = t_x(x)
        x = torch.FloatTensor(x)
        return self.forward(x.to(device)).detach().cpu().numpy()

    def to_float(self, x, t_x=None):
        return self.to_numpy(x, t_x=t_x).item()

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = self.forward(state.to(device))
        action = action.detach().cpu().numpy()
        if self.is_stochastic:
            action = multivariate_normal.rvs(action, self._sigma, 1)
        return action

    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path + "/policy")

    def load(self, path):
        self.load_state_dict(torch.load(
            path + "policy", map_location=device))
