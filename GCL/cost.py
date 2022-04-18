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
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
