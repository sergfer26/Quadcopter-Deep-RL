import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from .utils import RolloutBuffer
from .params import PARAMS_PPO as params


class PPOAgent:

    def __init__(self, env, hidden_sizes=params['hidden_sizes'], \
        learning_rate_actor=params['learning_rate_actor'], \
        learning_rate_critic=['learning_rate_critic'], \
        gamma=['gamma'], K_epochs=['K_epochs'], eps_clip=['eps_clip'], \
        action_std_init=['action_std_init']):

        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

