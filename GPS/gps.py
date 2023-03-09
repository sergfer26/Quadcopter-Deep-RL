import torch
import pathlib
import numpy as np
import multiprocessing as mp
from scipy.stats import multivariate_normal
from torch import optim
from copy import deepcopy
from os.path import exists
# from functools import partial
from .utils import OfflineCost  # , OnlineCost
# from ilqr import RecedingHorizonController
from .params import PARAMS_LQG
from GPS.utils import ContinuousDynamics
from .controller import OfflineController  # , OnlineController, OnlineMPC
from torch.distributions.multivariate_normal import _batch_mahalanobis
from .utils import nearestPD


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class GPS:

    def __init__(self, env, policy,
                 dynamics_kwargs,
                 cost, cost_terminal=None,
                 t_x=None, t_u=None,
                 inv_t_x=None, inv_t_u=None,
                 N=3, M=2, eta=1e-3, nu=1e-2,
                 lamb=1e-3, alpha_lamb=1e-1,
                 learning_rate=0.01, kl_step=200,
                 per_kl=.1, known_dynamics=False,
                 mask=None):
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
        self.kl_step = kl_step
        self.per_kl = per_kl

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
                                T=T,
                                known_dynamics=known_dynamics)  # ,
        # u_bound=u_bound)

        self.policy = policy
        self.policy_sigma = np.identity(self.n_u)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate)

        self.mask = mask

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
        # if len(xs.shape) == 4:  # (N, M, T, n_x)
        #     arg_x = 'NMTux,NMTx ->NMTu'
        #     arg_k = 'NMT,NMTu ->NMTu'
        # elif len(xs.shape) == 3:  # (N, T, n_x)
        #     arg_x = 'NTux,NTx ->NTu'
        #     arg_k = 'NT,NTu ->NTu'
        # elif len(xs.shape) == 2:  # (T, x)
        #     arg_x = 'Tux,Tx -> Tu'
        #     arg_k = 'T,Tu ->Tu'
        us_mean = np.einsum('NTux, NMTx -> NMTu', K, xs - nominal_xs)
        us_mean += np.expand_dims(nominal_us, axis=1)
        us_mean += np.expand_dims(np.einsum('N, NTu-> NTu', alpha, k), axis=1)
        return us_mean

    def cov_policy(self, C):
        if len(C.shape) == 5:
            axis = (0, 1, 2)
        elif len(C.shape) == 4:
            axis = (0, 1)
        Q = np.linalg.inv(C)
        C_new = nearestPD(np.linalg.inv(np.mean(Q, axis=axis)))
        # C_new += PARAMS_LQG['cov_reg'] * np.identity(C.shape[-1])
        return C_new

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
        # Check if is symetric positive definite:
        eigvals = np.linalg.eigvals(C)
        eigvals_sigma = np.linalg.eigvals(sigma)
        raise_error = False
        if not np.apply_along_axis(np.greater, -1, eigvals, np.zeros(self.n_u)).all():
            raise_error = True
            error = 'C has negative eigen values'
        if not np.apply_along_axis(np.greater, -1, eigvals_sigma, np.zeros(self.n_u)).all():
            raise_error = True
            error = 'sigma has negative eigen values: \n {sigma}'

        if raise_error:
            raise ValueError(error)

        if len(states.shape) == 4:
            arg1 = 'NMT,NT-> NMT'
            arg2 = 'NMTu,NTu-> NMT'
        elif len(states.shape) == 3:
            arg1 = 'NT,NT-> NT'
            arg2 = 'NTu,NTu-> NT'
        else:
            print('states de dimensión {states.shape} no es compatible')

        if len(C.shape) == 4:
            C = np.expand_dims(C, axis=1)
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
        K = np.empty((self.N, self.T, self.n_u, self.n_x))
        k = np.empty((self.N, self.T, self.n_u))
        C = np.empty((self.N, self.T, self.n_u, self.n_u))
        nominal_xs = np.empty((self.N, self.T + 1, self.n_x))
        nominal_us = np.empty((self.N, self.T, self.n_u))
        alphas = np.empty(self.N)
        xs = np.empty((self.N, self.M, self.T + 1, self.n_x))
        us = np.empty((self.N, self.M, self.T, self.n_u))
        for i in range(self.N):
            file = np.load(path + f'control_{i}.npz')
            K[i] = file['K']
            k[i] = file['k']
            C[i] = file['C'] + PARAMS_LQG['cov_reg'] * np.identity(self.n_u)
            nominal_xs[i] = file['xs']
            nominal_us[i] = file['us']
            rollouts = np.load(path + f'rollouts_{i}.npz')
            xs[i] = rollouts['xs']
            us[i] = rollouts['us']
            alphas[i] = file['alpha']
        nominal_xs = nominal_xs[:, :, :-1]
        xs = xs[:, :, :-1]
        return K, k, C, nominal_xs, nominal_us, xs, us, alphas

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
                x0 = self.env.observation_space.sample()
                cost_kwargs = deepcopy(self.cost_kwargs)
                cost_kwargs['lamb'] = self.lamb[i]
                cost_kwargs['nu'] = self.nu[i]
                cost_kwargs['eta'] = self.eta[i]
                p = mp.Process(target=fit_ilqg,
                               args=(x0,
                                     self.kl_step,
                                     self.policy,
                                     cost_kwargs,
                                     self.dynamics_kwargs,
                                     i,
                                     self.T,
                                     self.M,
                                     path,
                                     self.t_x,
                                     self.inv_t_u,
                                     self.policy._sigma,
                                     self.mask
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
                    self.kl_step,
                    self.policy,
                    cost_kwargs,
                    self.dynamics_kwargs,
                    i,
                    self.T,
                    self.M,
                    path,
                    self.t_x,
                    self.inv_t_u,
                    self.policy._sigma,
                    self.mask
                    )
            fit_ilqg(*args)
        # 1.1 update eta
        self.eta = self._update_eta(path)
        # 1.2 Loading fitted parameters
        (K, k, C, nominal_xs, nominal_us,
         xs, us, alphas) = self._load_fitted_lqg(path)
        if np.isnan(K).any():
            raise ValueError('K has NaNs')
        if np.isnan(k).any():
            raise ValueError('k has NaNs')
        if np.isnan(C).any():
            raise ValueError('C has NaNs')
        if np.isnan(self.eta).any():
            raise ValueError('eta has NaNs')
        if np.isnan(nominal_us).any():
            raise ValueError('nominal us has NaNs')
        if np.isnan(nominal_xs).any():
            raise ValueError('nominal xs has NaNs')
        if np.isnan(xs).any():
            raise ValueError('xs has NaNs')
        if np.isnan(us).any():
            raise ValueError('us has NaNs')
        # 1.3 Loading simulations
        # 2. Policy fitting
        us_mean = self.mean_control(xs, nominal_xs, nominal_us, K, k, alphas)
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

        # 4. Update kl_step
        self.kl_step = self.kl_step * (1 - self.per_kl)
        return loss, div


def fit_ilqg(x0, kl_step, policy, cost_kwargs, dynamics_kwargs, i, T, M,
             path='', t_x=None, inv_t_u=None, policy_sigma=None, mask=None):
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
    n_x = dynamics_kwargs['n_x']
    n_u = dynamics_kwargs['n_u']
    if not isinstance(policy_sigma, np.ndarray):
        policy_sigma = np.identity(dynamics_kwargs['n_u'])
    dynamics = ContinuousDynamics(**dynamics_kwargs)

    # ###### Instancias control iLQG #######
    cost = OfflineCost(**cost_kwargs)
    cost.update_policy(policy=policy, t_x=t_x,
                       inv_t_u=inv_t_u, cov=policy_sigma)
    control = OfflineController(dynamics, cost, T,
                                known_dynamics=cost_kwargs['known_dynamics'])

    # Actualización de parametros de control para costo
    file_name = f'control_{i}.npz'
    if exists(path + file_name):
        # file_name = file_name if exists(file_name) else 'control.npz'
        control.load(path, file_name)
    else:
        # 'results_offline/23_02_01_13_30/'
        # 'results_offline/23_02_15_09_14/' -> para 2 seg
        # 'results_offline/23_02_16_13_50/'
        # control.load('results_ilqr/23_02_16_23_21/')
        # control.load('results_offline/23_02_15_09_14/')
        control.load('results_offline/23_02_16_13_50/')
    # También puede actualizar eta
    cost.update_control(control)
    is_stochastic = control.is_stochastic
    control.is_stochastic = False
    xs, us = control.rollout(x0)
    control.is_stochastic = is_stochastic
    # control.fit_control(xs[0], us_init)
    control.x0, control.us_init = x0, us
    control.optimize(kl_step, cost.eta)
    control.save(path=path, file_name=f'control_{i}.npz')

    states = np.empty((M, T + 1, n_x))
    actions = np.empty((M, T, n_u))
    control.is_stochastic = True
    x0_samples = multivariate_normal.rvs(mean=x0, cov=np.identity(n_x), size=M)
    if isinstance(mask, np.ndarray):
        x0_samples[:, mask] = x0[mask]
    for r in range(M):
        xs, us = control.rollout(x0_samples[r])
        states[r] = xs
        actions[r] = us

    np.savez(path + f'rollouts_{i}.npz',
             xs=states,
             us=actions
             )

    # Ajuste MPC
