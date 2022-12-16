import torch
import pandas as pd
import numpy as np
import torch.autograd
import pathlib
from .cost import CostNN
from .utils import Memory


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class GCL:

    def __init__(self, memory: Memory, ilqr, env,
                 tx=None, tu=None, inv_tx=None, inv_tu=None, x_dim=None, y_dim=None):
        # self.expert = expert
        self.ilqr = ilqr
        self.env = env
        self.state_dim = memory.state_dim
        self.action_dim = memory.action_dim
        self.steps = env.steps
        self.tx = tx
        self.tu = tu
        self.inv_tx = inv_tx
        self.inv_tu = inv_tu
        self.cost = CostNN(self.state_dim, self.action_dim)
        self.cost_optimizer = torch.optim.Adam(
            self.cost.parameters(), 1e-2, weight_decay=1e-4)
        self.memory_demo = memory
        self.memory_samp = Memory(
            int(1e6),
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            T=env.steps - 1
        )

    def set_cost(self, cost):
        self.cost = cost
        self.cost_optimizer = torch.optim.Adam(
            self.cost.parameters(), 1e-2, weight_decay=1e-4)

    def distribution_traj(self, traj):
        if isinstance(traj, list):
            states, actions, next_states = traj
        elif isinstance(traj, np.ndarray):
            states = traj[:, : self.state_dim]
            actions = traj[:, self.state_dim: self.state_dim + self.action_dim]
            index = self.state_dim + self.action_dim + self.state_dim
            next_states = traj[:, self.state_dim + self.action_dim: index]
        steps, _ = states.shape
        if callable(self.inv_tx):
            states = np.apply_along_axis(self.inv_tx, 1, states)
            next_states = np.apply_along_axis(self.inv_tx, 1, next_states)
        if callable(self.inv_tu):
            actions = np.apply_along_axis(self.inv_tu, 1, actions)
        pdf = self.ilqr.get_prob_state(state=states[0, :], t=0)
        for t in range(steps):
            s, a, n_s = states[t, :], actions[t, :], next_states[t, :]
            pdf *= self.ilqr.get_prob_action(state=s, action=a, t=t)
            # state=s, action=a, new_state=n_s)
            pdf *= self.ilqr.get_prob_state(state=n_s, t=t)
        return pdf

    def get_prob_action(self, state, action, t=0):
        if callable(self.inv_tx):
            state = self.inv_tx(state)
        if callable(self.inv_tu):
            action = self.inv_tu(action)
        return self.ilqr.get_prob_action(state=state, action=action, t=t)

    def get_probs_trajs(self, actions, states):
        '''
        Calcula la expresión \{q(\tau)_{j}\}_{j} de
        la verosimilitud.

        Argumentos 
        ----------
        actions : `np.ndarray`
            Arreglo con dimensiones (n, self.steps-1, self.action_dim).
        states : `np.ndarray`
            Arreglo con dimensiones (n, self.steps-1, self.state_dim).

        Retorno
        -------
        probs : `np.ndarray`
            Arreglo con las dimensiones (n,)
        '''
        n_x = self.state_dim
        def prob_fun(z, i): return self.get_prob_action(z[:n_x], z[n_x:], t=i)
        array_probs_t = np.empty_like(states[:, :, 0])
        for t in range(states.shape[1]):
            x_u = np.concatenate([states[:, t], actions[:, t]], axis=1)
            array_probs_t[:, t] = np.apply_along_axis(
                lambda z: prob_fun(z, t), -1, x_u)
        return np.prod(array_probs_t, axis=1)

    def fit_dynamics(self, next_states):
        if callable(self.inv_tx):
            next_states = np.apply_along_axis(self.inv_tx, -1, next_states)
        self.ilqr.fit_dynamics(next_states)

    def train_cost(self, demo_size, samp_size):
        self.cost.train()
        # for _ in range(REWARD_FUNCTION_UPDATE):
        states_demo, actions_demo, new_states_demo = self.memory_demo.sample(demo_size)[
            0:-1]
        states_samp, actions_samp, new_states_samp = self.memory_samp.sample(samp_size)[
            0:-1]
        # states_traj, actions_traj, new_states_traj = self.sample_rollouts(
        #     TRAJ_SIZE)
        states_samp = np.vstack([states_samp, states_demo])  # states_traj
        actions_samp = np.vstack(
            [actions_samp, actions_demo])  # states_traj
        new_states_samp = np.vstack([new_states_samp, new_states_demo])
        probs = self.get_probs_trajs(states_samp, actions_samp)
        # states_samp = states_samp.reshape(
        #     states_samp.shape[0] * states_samp.shape[1], -1)
        # actions_samp = actions_samp.reshape(
        #     actions_samp.shape[0] * actions_samp.shape[1], -1)
        # states_demo = states_demo.reshape(
        #     states_demo.shape[0] * states_demo.shape[1], -1)
        # actions_demo = actions_demo.reshape(
        #     actions_demo.shape[0] * actions_demo.shape[1], -1)
        # Reducing from float64 to float32 for making computaton faster
        states_tensor = torch.tensor(states_samp, dtype=torch.float32)
        actions_tensor = torch.tensor(actions_samp, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        states_demo_tensor = torch.tensor(states_demo, dtype=torch.float32)
        actions_demo_tensor = torch.tensor(
            actions_demo, dtype=torch.float32)
        cost_samp = self.cost(states_tensor, actions_tensor)
        cost_demo = self.cost(states_demo_tensor, actions_demo_tensor)
        # probs = 0.1  # llamar a código agente ilqr
        loss_IOC = torch.mean(cost_demo.sum(1)) + \
            torch.log(torch.mean(torch.exp(-cost_samp.sum(1)) /
                      (probs.unsqueeze(dim=-1)+1e-4)))
        self.cost_optimizer.zero_grad()
        loss_IOC.backward()
        self.cost_optimizer.step()
        return loss_IOC.detach().item()
        # return loss_IOC.detach().item()

    def save(self, path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.cost.state_dict(), path + "/cost")

    def load(self, path):
        self.critic.load_state_dict(torch.load(
            path + "cost"))
