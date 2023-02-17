import torch
import pathlib
import numpy as np
import multiprocessing as mp

from torch import optim
from copy import deepcopy
from os.path import exists
from functools import partial
from .utils import OnlineCost, OfflineCost
# from ilqr import RecedingHorizonController
from GPS.utils import ContinuousDynamics
from .params import PARAMS_OFFLINE, PARAMS_ONLINE
from .controller import OnlineController, OfflineController, OnlineMPC
from torch.distributions.multivariate_normal import _batch_mahalanobis

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
                 N=3, M=2, eta=1e-3, nu=1e-3,
                 lamb=1e-3, alpha_lamb=1e-1,
                 learning_rate=0.01):
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
        eta : `float`
            Valor inicial de eta
        nu : `float`
            Valor inicial de nu.
        lamb : `float`
            Valor inicial de lambda
        '''
        T = env.steps - 1
        self.N = N  # Number of fitted iLQG controlllers.
        self.M = M  # Number of fitted MPC controllers per iLQG c.
        self.T = T  # Number of steps done in a simulation.

        self.n_x = env.observation_space.shape[0]  # State dimention.
        self.n_u = env.action_space.shape[0]       # Action dimention.

        # Transformation for control arguments
        self.t_x = t_x  # x_t -> o_t
        self.t_u = t_u  # u_t -> a_t
        self.inv_t_u = inv_t_u  # a_t -> u_t
        self.inv_t_x = inv_t_x  # o_t -> x_t

        # Instances for simulation.
        self.env = env

        # Lagrange's multipliers
        self.lamb = lamb * np.ones((N, T, self.n_u))   # contraint weight.
        # learning rate for lamb.
        self.alpha_lamb = alpha_lamb
        self.nu = nu * np.ones((N, T))               # KL-div weight.
        # old traj weight for iLQG.
        self.eta = eta * np.ones(N)

        # ilQG instances.
        if not callable(cost_terminal):
            def cost_terminal(x, i): return cost(x, np.zeros(self.n_u), i)

        self.dynamics_kwargs = dynamics_kwargs
        self.cost_kwargs = dict(cost=cost,
                                l_terminal=cost_terminal,
                                n_x=self.n_x,
                                n_u=self.n_u,
                                eta=eta,
                                lamb=lamb * np.ones(self.n_u),
                                nu=nu * np.ones(T),
                                T=T)  # ,
        # u_bound=u_bound)

        self.policy = policy
        self.policy_sigma = np.identity(self.n_u)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate)

        # Multiprocessing instances.
        self.manager = mp.Manager()
        self.buffer = self.manager.dict()

    def mean_control(self, xs, nominal_xs, nominal_us, K, k, alpha):
        '''
        Calcula la siguiente expresión
            u_s = \hat{u_s} + \alpha k + K (x_s - \hat{x_s})

        Argumentos
        ----------
        xs : `np.ndarray`
            Conjunto de trayectorias de estados con dimensiones (B, n_x).
        nominal_xs : `np.ndarray`
            Conjunto de trayectorias nominales de estado con
            dimensiones (B, n_x).
        nominal_us : `np.ndarray`
            Conjunto de trayectorias nominales de acción con
            dimensiones (B, n_u).
        K : `np.ndarray`
            Conjunto de parámetros para estado con dimensiones (B, n_u, n_x).
        k : `np.ndarray`
            Conjunto de parámetros para acción con dimensiones (B, n_u).
        alpha : `np.ndarray`
            Conjunto de parámetro de regularización con dimensiones (B,).

        Donde B representa la dimensión del batch y puede tener
        dimensión: 3, 2, 1.

        Retorno
        -------
        us_mean : `np.ndarray`
            Conjunto de trayectorias resultantes de acciones con
            dimensión (B, n_u).
        '''
        if len(xs.shape) == 4:  # (N, M, T, n_x)
            arg_x = 'NMTux,NMTx ->NMTu'
            arg_k = 'NMT,NMTu ->NMTu'
        elif len(xs.shape) == 3:  # (N, T, n_x)
            arg_x = 'NTux,NTx ->NTu'
            arg_k = 'NT,NTu ->NTu'
        elif len(xs.shape) == 2:  # (T, x)
            arg_x = 'Tux,Tx -> Tu'
            arg_k = 'T,Tu ->Tu'
        us_mean = np.einsum(arg_x, K, xs - nominal_xs)
        us_mean += nominal_us
        us_mean += np.einsum(arg_k, alpha, k)
        return us_mean

    def cov_policy(self, C):
        Q = np.linalg.inv(C)
        return np.linalg.inv(np.mean(Q, axis=(0, 1, 2)))

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
        if len(states.shape) == 4:
            arg1 = 'NMT,NT-> NMT'
            arg2 = 'NMTu,NTu-> NMT'
        elif len(states.shape) == 3:
            arg1 = 'NT,NT-> NT'
            arg2 = 'NMTu,NTu-> NMT'
        else:
            print('states de dimensión {states.shape} no es compatible')
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
        loss = torch.einsum(arg1, loss, nu)
        # reg = actions * lamb
        loss += torch.einsum(arg2, actions, lamb)  # reg.sum(dim=-1)
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

    def _update_eta(self, path):
        eta = np.empty(self.N)
        for i in range(self.N):
            file = np.load(path + f'control_{i}.npz')
            eta[i] = file['eta']
        return eta

    def _load_fitted_lqg(self, path):
        '''
        Retornos
        --------
        K : `np.ndarray`
            Dimensión (N, M, T, n_u, n_x).
        k : `np.ndarray`
            Dimensión (N, M, T, n_u).
        C : `np.ndarray`
            Dimensión (N, M, T, n_u, n_u)
        xs : `np.ndarray`
            Dimensión (N, M, T, n_x)
        us : `np.ndarray`
            Dimensión (N, M, T, n_u)
        xs_old : `np.ndarray`
            Dimensión (N, M, T, n_x)
        us_old : `np.ndarray`
            Dimensión (N, M, T, n_u)
        alphas : `np.ndarray`
            Dimensión (N, M, T)
        '''
        K = np.empty((self.N, self.M, self.T, self.n_u, self.n_x))
        k = np.empty((self.N, self.M, self.T, self.n_u))
        C = np.empty((self.N, self.M, self.T, self.n_u, self.n_u))
        xs = np.empty((self.N, self.M, self.T + 1, self.n_x))
        us = np.empty((self.N, self.M, self.T, self.n_u))
        alphas = np.empty((self.N, self.M, self.T))
        xs_old = np.empty_like(xs)
        us_old = np.empty_like(us)
        for i in range(self.N):
            for j in range(self.M):
                file = np.load(path + f'mpc_control_{i}_{j}.npz')
                K[i, j] = file['K']
                k[i, j] = file['k']
                C[i, j] = file['C']
                xs[i, j] = file['xs']
                us[i, j] = file['us']
                xs_old[i, j] = file['xs_old']
                us_old[i, j] = file['us_old']
                alphas[i, j] = file['alpha']
        xs = xs[:, :, :-1]
        xs_old = xs_old[:, :, :-1]
        return K, k, C, xs, us, xs_old, us_old, alphas

    def _update_lamb(self, us_policy, us_control):
        new_lamb = self.lamb
        mean_policy = np.mean(us_policy, axis=1)
        mean_control = np.mean(us_control, axis=1)
        u = mean_policy - mean_control
        new_lamb += self.alpha_lamb * np.einsum('NT, NTu -> NTu', self.nu, u)
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

        # 1. Control fitting
        if self.N > 1:
            processes = list()
            for i in range(self.N):
                x0 = [self.env.observation_space.sample()
                      for i in range(self.M)]
                cost_kwargs = deepcopy(self.cost_kwargs)
                cost_kwargs['lamb'] = self.lamb[i]
                cost_kwargs['nu'] = self.nu[i]
                cost_kwargs['eta'] = self.eta[i]
                p = mp.Process(target=fit_ilqg,
                               args=(x0,
                                     self.policy,
                                     cost_kwargs,
                                     self.dynamics_kwargs,
                                     i,
                                     self.T,
                                     self.M,
                                     path,
                                     self.t_x,
                                     self.inv_t_u,
                                     self.policy._sigma
                                     )
                               )
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        else:
            i = 0
            cost_kwargs = deepcopy(self.cost_kwargs)
            cost_kwargs['lamb'] = self.lamb[i]
            cost_kwargs['nu'] = self.nu[i]
            x0 = self.env.observation_space.sample()
            args = (x0,
                    self.policy,
                    cost_kwargs,
                    self.dynamics_kwargs,
                    i,
                    self.T,
                    self.M,
                    path,
                    self.t_x,
                    self.inv_t_u,
                    self.policy._sigma
                    )
            fit_ilqg(*args)
        # 1.1 update eta
        self.eta = self._update_eta(path)
        # 1.2 Loading fitted parameters
        K, k, C, xs, us, xs_old, us_old, alphas = self._load_fitted_lqg(path)
        # 1.3 Loading simulations
        # 2. Policy fitting
        us_mean = self.mean_control(xs, xs_old, us_old, K, k, alphas)
        self.policy._sigma = self.cov_policy(C)
        if callable(self.t_x):
            xs = np.apply_along_axis(self.t_x, -1, xs)  # x_t -> o_t

        loss, div = self.policy_loss(xs, us_mean, C, self.policy._sigma)
        div = div.detach().cpu().numpy()
        mean_div = np.mean(div, axis=1)  # (N, T)

        # 2.1 Update policy network
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        # 3. Update Langrange's multipliers
        # x_t -> o_t
        us_policy = self.policy.get_action(xs)
        if callable(self.inv_t_u):  # a_t -> u_t
            us_policy = np.apply_along_axis(self.inv_t_u, -1, us_policy)
        self.lamb = self._update_lamb(us_policy, us)
        self.nu = self._update_nu(mean_div)
        loss = loss.detach().cpu().item()

        return loss, div


def fit_ilqg(x0, policy, cost_kwargs, dynamics_kwargs, i, T, M,
             path='', t_x=None, inv_t_u=None, policy_sigma=None):
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
        policy_sigma = np.identity(dynamics_kwargs['n_u'])
    dynamics = ContinuousDynamics(**dynamics_kwargs)

    # ###### Instancias control iLQG #######
    cost = OfflineCost(**cost_kwargs)
    cost.mean_policy = lambda x: policy.to_numpy(x, t_x=t_x, t_u=inv_t_u)
    cost.cov_policy = policy_sigma
    control = OfflineController(dynamics, cost, T)

    # Actualización de parametros de control para costo
    file_name = f'control_{i}.npz'
    if exists(path + file_name):
        # file_name = file_name if exists(file_name) else 'control.npz'
        control.load(path, file_name)
    else:
        # 'results_offline/23_02_01_13_30/'
        # 'results_offline/23_02_15_09_14/' -> para 2 seg
        # 'results_offline/23_02_16_13_50/'
        control.load('results_offline/23_02_16_13_50/')
    # También actualiza eta
    cost.update_control(control)

    xs, us = control.rollout(np.zeros(dynamics_kwargs['n_x']))

    # control.fit_control(xs[0], us_init)
    control.x0, control.us_init = xs[0], us
    control.optimize(KL_STEP, cost.eta)
    control.save(path=path, file_name=f'control_{i}.npz')
    nu = cost_kwargs['nu']
    lamb = cost_kwargs['lamb']
    if M > 1:
        with mp.Pool(processes=M) as pool:
            pool.map(partial(_fit_child, policy, policy_sigma, t_x, inv_t_u,
                             x0, T, nu, lamb, dynamics_kwargs,
                             path, i), range(M))
            pool.close()
            pool.join()
    else:
        _fit_child(policy, policy_sigma, t_x, inv_t_u, x0,
                   T, nu, lamb, dynamics_kwargs, path, i, 0)


def _fit_child(policy, sigma, t_x, inv_t_u, x0, T, nu, lamb,
               dynamics_kwargs, path, i, j):
    if isinstance(x0, list):
        x0 = x0[j]
    dynamics = ContinuousDynamics(**dynamics_kwargs)
    control = OfflineController(dynamics, None, T)
    control.load(path, file_name=f'control_{i}.npz')

    # Creación de instancias control MPC-iLQG
    n_x = dynamics._state_size
    n_u = dynamics._action_size
    cost = OnlineCost(n_x, n_u, control, nu=nu,
                      lamb=lamb, F=PARAMS_ONLINE['F'])
    cost.mean_policy = lambda x: policy.to_numpy(x, t_x=t_x, t_u=inv_t_u)
    cost.cov_policy = sigma
    mpc_control = OnlineController(dynamics, cost, T)
    # agent = RecedingHorizonController(x0, mpc_control)
    agent = OnlineMPC(x0, mpc_control)
    control.is_stochastic = False
    us_init = control.rollout(x0)[1]
    horizon = PARAMS_ONLINE['step_size']
    traj = agent.control(us_init,
                         step_size=horizon,
                         initial_n_iterations=50,
                         subsequent_n_iterations=25)
    states = np.empty((T + 1, n_x))
    actions = np.empty((T, n_u))
    xs_old = np.empty_like(states)
    us_old = np.empty_like(actions)
    C = np.empty_like(control._C)
    K = np.empty_like(control._K)
    k = np.empty_like(control._k)
    alpha = np.empty_like(control._nominal_us)
    r = 0
    for t in range(mpc_control.N // horizon):
        xs, us = next(traj)
        C[t: t + horizon] = mpc_control._C[:horizon]
        K[t: t + horizon] = mpc_control._K[:horizon]
        alpha[t: t + horizon] = mpc_control.alpha
        k[t: t + horizon] = mpc_control._k[:horizon]
        states[r: r + horizon + 1] = xs
        actions[r: r + horizon] = us
        xs_old[r: r + horizon + 1] = mpc_control._xs[:horizon+1]
        us_old[r: r + horizon] = mpc_control._us[:horizon]
        r += horizon

    mpc_control._C = C
    mpc_control._K = K
    mpc_control._k = k
    mpc_control.alpha = alpha
    mpc_control._nominal_us = states
    mpc_control._nominal_xs = actions
    mpc_control._xs = xs_old
    mpc_control._us = us_old
    mpc_control.save(path, f'mpc_control_{i}_{j}.npz')

    # Ajuste MPC
