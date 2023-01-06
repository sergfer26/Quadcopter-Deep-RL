from torch.distributions.multivariate_normal import _batch_mahalanobis
from .utils import Memory
import torch
import numpy as np
from torch.distributions import MultivariateNormal
import torch.optim as optim


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class GPS:

    def __init__(self, n_x, n_u, policy, T=375, N=6, M=4, memory_size=1000,
                 learning_rate=0.01):
        self.n_x = n_x
        self.n_u = n_u

        self.memory = Memory(memory_size, n_u, n_x, T=T)
        self.policy = policy
        self.lamb = np.ones((N, T, n_u))
        self.nu = np.ones((N, T))
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate)

    def policy_loss(self, states, actions, C, sigma):
        '''
        Cálculo de la función de pérdida, ecuación [2].

        Argumentos
        ----------
        states : `np.ndarray`
            Estados con dimensiones `(b, n_x)`.
        actions : `np.ndarray`
            Acciones con dimensiones `(b, n_u)`.
        C : `np.ndarray`
            Matriz de covarianza de acciones de
            control con dimensiones `(b, n_u, n_u)`.
        sigma : `np.ndarray`
            Matriz de covarianza de acciones de
            pólitica con dimensiones `(b, n_u, n_u)`.

        Donde `b` representa el batch size, puede ser un 
        número entero o una tupla de números enteros.

        Retornos
        --------
        loss : `(b)`
        '''
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        C = torch.FloatTensor(C).to(device)
        sigma = torch.FloatTensor(sigma).to(device)
        lamb = torch.FloatTensor(self.lamb).to(device)
        # C = torch.FloatTensor(control._C).to(device)
        policy_actions = self.policy(states)
        loss = 0.0
        loss += 0.5 * _batch_mahalanobis(C, policy_actions - actions)
        # calculo de la traza
        loss -= 0.5 * \
            torch.diagonal(torch.matmul(C, sigma),
                           dim1=-2, dim2=-1).sum(-1)
        # loss -= 0.5 * torch.einsum('bii->b', C @ self.policy.sigma)
        loss += 0.5 * torch.log(torch.linalg.det(sigma))
        dkl = loss
        loss = torch.mul(loss.mT.T, self.nu.T)
        reg = actions * lamb
        loss += reg.sum(dim=-1)
        return loss, dkl

    def fit_policy(self, states, actions):
        pass

    def fit_lqg()
