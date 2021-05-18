import torch
import pathlib
import copy
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from .utils import RolloutBuffer
from .models import Actor, Critic
from .params import PARAMS_PPO as params


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class ActorCrtic(object):
    def __init__(self, h_sizes, num_actions, action_std_init):
        self.num_actions = num_actions
        self.action_var = torch.full(
            (num_actions,), action_std_init * action_std_init).to(device)

        self.actor = Actor(h_sizes, self.num_actions)
        self.critic = Critic(h_sizes)
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(
            (self.num_actions,), new_action_std * new_action_std).to(device)

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action = torch.clamp(action, min=-1.0, max=1.0)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOagent:

    def __init__(self, env, hidden_sizes=params['hidden_sizes'],
                 actor_learning_rate=params['actor_learning_rate'],
                 critic_learning_rate=params['critic_learning_rate'],
                 gamma=params['gamma'], K_epochs=params['K_epochs'],
                 eps_clip=params['eps_clip'],
                 action_std_init=params['action_std_init'],
                 action_std_decay_rate=params['action_std_decay_rate'],
                 min_action_std=params['min_action_std']):

        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        hidden_sizes.insert(0, self.num_states)

        # Networks
        self.policy = ActorCrtic(
            hidden_sizes, self.num_actions, self.action_std)
        self.old_policy = ActorCrtic(
            hidden_sizes, self.num_actions, self.action_std)

        for target_param, param in zip(self.old_policy.actor.parameters(), self.policy.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.old_policy.critic.parameters(), self.policy.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.buffer = RolloutBuffer()
        self.criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(
            self.policy.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.old_policy.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self):
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= self.min_action_std):
            self.action_std = self.min_action_std
            print("setting actor output action_std to min_action_std : ",
                  self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / \
        #    (rewards.std() + 1e-7)  # Revisar esta linea

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.criterion(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # Copy new weights into old policy
        self.old_policy.actor.load_state_dict(self.policy.actor.state_dict())
        self.old_policy.critic.load_state_dict(self.policy.critic.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.critic.state_dict(), path + "/critic")
        torch.save(self.critic_optimizer.state_dict(),
                   path + "/critic_optimizer")
        torch.save(self.policy.actor.state_dict(), path + "/actor")
        torch.save(self.actor_optimizer.state_dict(),
                   path + "/actor_optimizer")

    def load(self, path):
        self.policy.critic.load_state_dict(torch.load(
            path + "/critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(
            path + "/critic_optimizer",  map_location=device))
        self.old_critic = copy.deepcopy(self.policy.critic)
        self.policy.actor.load_state_dict(torch.load(
            path + "/actor",  map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(
            path + "/actor_optimizer",  map_location=device))
        self.old_actor = copy.deepcopy(self.policy.actor)
