from torch.distributions.multivariate_normal import _batch_mahalanobis
from .utils import OfflineCost
from os.path import exists
import torch
import pathlib
import numpy as np
import torch.optim as optim
from copy import deepcopy
from .controller import iLQG, DummyControl
import multiprocessing as mp
from functools import partial
from GPS.utils import ContinuousDynamics, rollout

from .params import PARAMS_OFFLINE

KL_STEP = PARAMS_OFFLINE['kl_step']

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class GPS:

    def __init__(self, env, policy,
                 dynamics_kwargs,
                 cost, cost_terminal=None,
                 t_x=None, t_u=None,
                 inv_t_x=None, inv_t_u=None,
                 N=2, M=3,  # memory_size=1000,
                 learning_rate=0.01, u_bound=None):
        '''
        env : `gym.Env`
            Entorno de simulación de gym.
        policy: `Policy(nn.Module)`
            Red neuronal artificial que actua como política.
        dynamics_kwargs : `utils.ContinuousDynamics (kwargs)`
            Argumentos de sistema dinámico de simulación.
        cost: callable
            Función de costos. Signature -> (x, u, i).
        terminal_cost : `callable`
            Función de costos teminal. Signature -> (x, i).
        rollout_function : `callable`
            Función de simulación. Signature -> (policy, env)
        t_x : `callable`
            o_t -> x_t
        t_u : `callable`
            a_t -> u_t
        inv_t_x : `callable`
            x_t -> o_t
        '''
        T = env.steps - 1
        self.N = N  # Number of fitted iLQG controlllers.
        self.M = M  # Number of simulations from iLQG controllers.
        self.T = T  # Number of steps done in a simulation.

        self.n_x = env.observation_space.shape[0]  # State dimention.
        self.n_u = env.action_space.shape[0]        # Action dimention.

        # Transformation for control arguments
        self.t_x = t_x  # x_t -> o_t
        self.t_u = t_u  # u_t -> a_t
        self.inv_t_u = inv_t_u  # a_t -> u_t
        self.inv_t_x = inv_t_x  # o_t -> x_t

        # Instances for simulation.
        self.env = env

        # Lagrange's multipliers
        self.lamb = np.ones((N, T, self.n_u))   # contraint weight.
        self.alpha_lamb = 0.1                   # learning rate for lamb.
        self.nu = 0.001 * np.ones((N, T))       # KL-div weight.
        # old traj weight for iLQG.
        self.eta = 1e-4 * np.ones(N)

        # ilQG instances.
        if not callable(cost_terminal):
            def cost_terminal(x, i): return cost(x, np.zeros(self.n_u), i)

        self.dynamics_kwargs = dynamics_kwargs
        self.cost_kwargs = dict(cost=cost,
                                l_terminal=cost_terminal,
                                n_x=self.n_x,
                                n_u=self.n_u,
                                eta=1e-4,
                                lamb=np.ones(self.n_u),
                                nu=np.ones(T),
                                T=T)  # ,
        # u_bound=u_bound)

        self.policy = policy
        self.policy_sigma = np.identity(self.n_u)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate)

        # Multiprocessing instances.
        self.manager = mp.Manager()
        self.buffer = self.manager.dict()

        # Control iLQG parameters.
        self._K = np.empty((N, T, self.n_u, self.n_x))
        self._k = np.empty((N, T, self.n_u))
        self._C = np.empty((N, T, self.n_u, self.n_u))
        self._nominal_xs = np.empty((N, T + 1, self.n_x))
        self._nominal_us = np.empty((N, T, self.n_u))
        self._alphas = np.empty(N)

    def mean_control(self, xs, i):
        nominal_us = self._nominal_us[i]
        x = xs - self._nominal_xs[i]
        us_mean = np.einsum('BNi,MBi ->MBN', self._K[i], x[:, :-1])
        us_mean += nominal_us
        us_mean += self._alphas[i] * self._k[i]
        return us_mean

    def cov_policy(self):
        Q = np.empty((self.N, self.n_u, self.n_u))
        for i in range(self.N):
            Q[i] = np.sum([np.linalg.inv(self._C[i, j])
                          for j in range(self.M)], axis=0)
        return np.sum(Q, axis=0) / (self.N * self.T)

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

        Donde `b` es número entero o tupla que representa el batch size.

        Retornos
        --------
        loss : `torch.Tensor`
            Salida de lagrangiano. Tensor de dimensión 0.
        kld : `torch.Tensor`
            Divergencia de Kullback-Leibler entre control y política
            en cada paso de tiempo. Tensor de dimensión `b = (N, T)`.
        '''
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        C = torch.FloatTensor(C).to(device)
        sigma = torch.FloatTensor(sigma).to(device)
        lamb = torch.FloatTensor(self.lamb).to(device)
        nu = torch.FloatTensor(self.nu).to(device)
        # C = torch.FloatTensor(control._C).to(device)
        policy_actions = self.inv_t_u(self.policy(states))

        loss = 0.0
        loss += 0.5 * _batch_mahalanobis(C, policy_actions - actions)
        # calculo de la traza
        loss -= 0.5 * \
            torch.diagonal(torch.matmul(C, sigma), dim1=-2, dim2=-1).sum(-1)
        # loss -= 0.5 * torch.einsum('bii->b', C @ self.policy.sigma)
        loss += 0.5 * torch.log(torch.linalg.det(sigma))
        kld = loss
        loss = loss * nu  # torch.mul(loss.mT.T, nu.T)
        # reg = actions * lamb
        loss += torch.einsum('ijk,ijk->ij', actions, lamb)  # reg.sum(dim=-1)
        loss = torch.sum(loss.flatten(), dim=0)
        # kld = torch.mean(kld, dim=0)
        return loss, kld

    def _load_buffer(self, path):
        xs = np.empty((self.N, self.M, self.T+1, self.n_x))
        us = np.empty((self.N, self.M, self.T, self.n_u))
        for i in range(self.N):
            for j in range(self.M):
                with np.load(path + f'traj_{i}_{j}.npz', allow_pickle=True) as file:
                    xs[i, j] = file['xs']
                    us[i, j] = file['us']
        return xs, us

    def _load_fitted_lqg(self, path):
        for i in range(self.N):
            file = np.load(path + f'control_{i}.npz')
            self._K[i] = file['K']
            self._k[i] = file['k']
            self._C[i] = file['C']
            self._nominal_xs[i] = file['xs']
            self._nominal_us[i] = file['us']
            self._alphas[i] = file['alpha']

    def _update_lamb(self, us_policy, us_control):
        new_lamb = self.lamb
        mean_policy = np.mean(us_policy, axis=1)
        mean_control = np.mean(us_control, axis=1)
        u = mean_policy - mean_control
        new_lamb += self.alpha_lamb * np.einsum('Bi, Biu -> Biu', self.nu, u)
        return new_lamb

    def _update_nu(self, div):
        '''
        div : `np.ndarray``
            La divergencia de KL entre la pólitica pi y el control p.
            Con dimensiones (N, T).
        '''
        nu = deepcopy(self.nu)
        # div = np.mean(div, axis=1)  # (N, T)
        mean_div = np.mean(div, axis=-1)  # (N,)
        std_div = np.std(div, axis=-1)
        mask = np.apply_along_axis(lambda x: np.greater(x, mean_div), 0, div)
        nu[mask] *= 2.0
        mask = np.apply_along_axis(lambda x: np.less(x, 2 * std_div), 0, div)
        nu[mask] /= 2.0
        return nu

    def update_policy(self, path):
        pathlib.Path(path + 'buffer/').mkdir(parents=True, exist_ok=True)
        path = path + 'buffer/'
        processes = list()
        # 1. Control fitting
        for i in range(self.N):
            cost_kwargs = deepcopy(self.cost_kwargs)
            cost_kwargs['lamb'] = self.lamb[i]
            cost_kwargs['nu'] = self.nu[i]
            cost_kwargs['eta'] = self.eta[i]
            p = mp.Process(target=fit_ilqg,
                           args=(self.env,
                                 self.policy,
                                 cost_kwargs,
                                 self.dynamics_kwargs,
                                 i,
                                 self.T,
                                 self.M,
                                 path,
                                 self.policy.env,
                                 self.t_x,
                                 self.inv_t_x,
                                 self.policy_sigma
                                 )
                           )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        # i = 0
        # cost_kwargs = deepcopy(self.cost_kwargs)
        # cost_kwargs['lamb'] = self.lamb[i]
        # cost_kwargs['nu'] = self.nu[i]
        # args = (self.env,
        #         self.policy,
        #         cost_kwargs,
        #         self.dynamics_kwargs,
        #         i,
        #         self.T,
        #         self.M,
        #         path,
        #         self.policy.env,
        #         self.inv_t_x,
        #         self.inv_t_u
        #         )
        # fit_ilqg(*args)
        # 1.1 update eta
        for i in range(self.N):
            file = np.load(path + f'control_{i}.npz')
            self.eta[i] = file['eta']

        # 1.2 Loading fitted parameters
        self._load_fitted_lqg(path)
        # 1.3 Loading simulations
        xs, us = self._load_buffer(path)
        # 2. Policy fitting
        us_mean = np.array([self.mean_control(xs[i], i)
                            for i in range(self.N)])
        self.policy_sigma = self.cov_policy()
        if callable(self.t_x):
            xs = np.apply_along_axis(self.t_x, -1, xs)  # x_t -> o_t
        total_loss = 0.0
        samples_div = np.empty((self.N, self.M, self.T))
        for j in range(self.M):
            x = xs[:, j, :-1]
            u = us_mean[:, j]
            loss, div = self.policy_loss(x, u, self._C, self.policy_sigma)
            total_loss += loss
            samples_div[:, j] = div.detach().cpu().numpy()
        mean_div = np.mean(samples_div, axis=1)  # (N, T)

        # 2.1 Update policy network
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        # 3. Update Langrange's multipliers
        # x_t -> o_t
        us_policy = self.policy.get_action(xs[:, :, :-1])
        if callable(self.inv_t_u):  # a_t -> u_t
            us_policy = np.apply_along_axis(self.inv_t_u, -1, us_policy)
        self.lamb = self._update_lamb(us_policy, us)
        self.nu = self._update_nu(mean_div)
        loss = loss.detach().cpu().item()

        # 4. Update parameters for cost
        self.cost_kwargs['nu'] = self.nu
        self.cost_kwargs['lamb'] = self.lamb

        return loss, div


def fit_ilqg(env, policy, cost_kwargs, dynamics_kwargs, i, T, M, path='',
             policy_env=None, t_x=None, inv_t_x=None, policy_sigma=None):
    '''
    i : int
        Indice de trayectoria producida por iLQG.
    T : int
        Número de pasos de la trayectoria.
    horizon : int
        Tamaño de la ventana de horizonte del MPC.
    M : int
        Indice de trayectoria producida por MPC.
    '''
    if not isinstance(policy_sigma, np.ndarray):
        policy_sigma = np.indentity(dynamics_kwargs['n_u'])
    low = env.observation_space.low
    high = env.observation_space.high
    steps = env.steps - 1
    dynamics = ContinuousDynamics(**dynamics_kwargs)

    # ###### Instancias control iLQG #######
    cost = OfflineCost(**cost_kwargs)
    cost.mean_policy = lambda x: policy.to_numpy(x, t_x=t_x)
    cost.cov_policy = policy_sigma
    control = iLQG(dynamics, cost, steps, low, high)

    # Actualización de parametros de control para costo
    file_name = f'control_{i}.npz'
    if exists(path + file_name):
        # file_name = file_name if exists(file_name) else 'control.npz'
        control.load(path, file_name)
    else:
        control.load('results_offline/23_02_01_13_30/')

    cost.update_control(control)

    if policy_env is None:
        policy_env = env
    xs, us_init, _ = rollout(policy, policy_env)
    if callable(inv_t_x):
        xs = np.apply_along_axis(inv_t_x, -1, xs)  # o_t -> x_t

    # control.fit_control(xs[0], us_init)
    control.x0, control.us_init = xs[0], us_init
    control.optimize(KL_STEP, cost.eta)[3]
    control.save(path=path, file_name=f'control_{i}.npz')
    with mp.Pool(processes=M) as pool:
        pool.map(partial(_fit_child, env,
                 T, path, i), range(M))
        pool.close()
        pool.join()


def _fit_child(env, T, path, i, j):

    file_name = f'control_{i}.npz'
    control = DummyControl(T, path, file_name)
    xs, us, _ = rollout(control, env)
    np.savez(path + f'traj_{i}_{j}.npz',
             xs=xs,
             us=us
             )
