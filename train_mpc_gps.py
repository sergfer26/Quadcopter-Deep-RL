import numpy as np
from GPS.controller import iLQRAgent
from GPS.utils import OfflineCost, FiniteDiffDynamics
from DDPG.models import Actor
from env import QuadcopterEnv
from dynamics import transform_x

n_x = 12
n_u = 4
env = QuadcopterEnv()
policy = Actor(n_x, n_u, [64, 64])
def cost(x, u, i): return - env.get_reward(x, u)
def cost_terminal(x, i): return - env.get_reward(x, np.zeros(4))
def mean_policy(x): return policy.to_float(x, t_x=transform_x)


class Policy:

    def __init__(self, n_x, n_u, hidden_sizes=[64, 64]):
        self.n_x = n_x
        self.n_u = n_u


cost_ilqr = OfflineCost(cost, cost_terminal, n_x, n_u,
                        eta=0.1, nu=0.001,
                        N=env.steps - 1, mean=mean_policy)
