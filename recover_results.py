import numpy as np
from simulation import n_rollouts
from env import QuadcopterEnv
from DDPG.utils import AgentEnv
from dynamics import inv_transform_x, transform_x
import pandas as pd
from params import PARAMS_DDPG

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
    def __init__(self, env, hidden_sizes):
        super(Policy, self).__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.env = env
        self.state_dim = state_dim

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
        self._C = 1e-1 * np.identity(action_dim)
        self.is_stochastic = False

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

    def to_numpy(self, x, t_x=None, t_u=None):
        if callable(t_x):
            x = t_x(x)
        x = torch.FloatTensor(x)
        out = self.forward(x.to(device)).detach().cpu().numpy()
        if callable(t_u):
            out = t_u(out)
        return out

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = self.forward(state.to(device))
        action = action.detach().cpu().numpy()
        if self.is_stochastic:
            action = multivariate_normal.rvs(action, self._C, 1)
        return action

    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path + "/policy")

    def load(self, path):
        self.load_state_dict(torch.load(
            path + "policy", map_location=device))


HIGH_RANGE = np.array(
    [.0, .0, .0, 1., 1., 1., .0, .0, .0, np.pi/64, np.pi/64, np.pi/64])
LOW_RANGE = - HIGH_RANGE


def _random_x0(x0, n, n_x: int = 12):
    size = (n, n_x)
    x = np.random.uniform(LOW_RANGE, HIGH_RANGE, size)
    return x + x0


def select_x0(x0: np.ndarray, n: int, return_indices: bool = False):
    N = x0.shape[0]
    if N > 1:
        indices = np.random.choice(N, n)
        states_init = x0[indices]
        states_init = np.apply_along_axis(_random_x0, -1, states_init, 1)
        states_init = np.squeeze(states_init, axis=1)
    else:
        states_init = _random_x0(x0, n)
    if return_indices:
        return states_init, indices
    return states_init


if __name__ == '__main__':

    M = 800
    x0 = np.load('states_init.npz')['states_init']
    env = other_env = AgentEnv(
        QuadcopterEnv(), tx=transform_x, inv_tx=inv_transform_x)
    hidden_sizes = PARAMS_DDPG['hidden_sizes']
    policy = Policy(env, hidden_sizes)
    policy_states = np.empty((7, M, 12))
    policy_costs = np.empty((7, M))
    array = np.load('results_gps/24_03_26_22_54/results.npz')
    policy_states[:-2] = array['policy_states']
    policy_costs[:-2] = array['policy_cost']
    for i in [-2, -1]:
        states_init, indices = select_x0(x0, M, return_indices=True)
        states_init = np.apply_along_axis(transform_x, -1, states_init)
        states, _, scores = n_rollouts(policy,
                                       env,
                                       M,
                                       t_x=inv_transform_x,
                                       states_init=states_init)

        policy_states[i] = states[:, -1]
        policy_costs[i] = scores[:, -1, 1]
        policy.load('results_gps/23_07_31_12_15/')

    data = np.apply_along_axis(np.linalg.norm, -1, policy_states)
    df = pd.DataFrame(data.T, columns=[
                      'iter 1', 'iter 2', 'iter 3', 'iter 4', 'iter 5', 'policy', 'policy trained'])
    quantiles = df.quantile([0.25, 0.5, 0.75, 0.90, 0.99])
    mean = df.mean()
    std_deviation = df.std()

    print("Standard deviation per column:")
    print(std_deviation)

    print("Quantiles per column:")
    print(quantiles)

    print("\nMean per column:")
    print(mean)

    np.savez('results.npz', states=policy_states, costs=policy_costs)
