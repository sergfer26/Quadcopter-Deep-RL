import torch
import torch.nn as nn


class CostNN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        out_features=1,
    ):
        super(CostNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.ReLU()
        )

    def forward(self, state, action):
        z = torch.concat([state, action], axis=-1)
        return self.net(z)

    def to_numpy(self, x, u, t_x=None, t_u=None):
        if callable(t_x):
            x = t_x(x)
        if callable(t_u):
            u = t_u(u)
        x = torch.FloatTensor(x)
        u = torch.FloatTensor(u)
        return self.forward(x, u).detach().numpy()

    def to_float(self, x, u, t_x=None, t_u=None):
        return self.to_numpy(x, u, t_x=t_x, t_u=t_u).item()
