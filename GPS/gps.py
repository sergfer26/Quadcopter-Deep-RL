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
from .params import PARAMS_LQG, PARAMS_OFFLINE
from .controller import OfflineController, iLQG
from torch.distributions.multivariate_normal import _batch_mahalanobis
from .utils import nearestPD, iLQR_Rollouts,  ContinuousDynamics
from torch.utils.data import DataLoader
from .convex_hull import ConvexSet, UniformSet, classifier, confidence_region
from typing import List, Union


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class GPS:

    def __init__(self, env, policy,
                 dynamics_kwargs,
                 cost, cost_terminal=None,
                 t_x=None, t_u=None,
                 inv_t_x=None, inv_t_u=None,
                 N: int = 3, M: int = 2,
                 eta: float = 1e-3,
                 nu: float = 1e-4,
                 lamb: float = 1e-9,
                 alpha_lamb: float = 1e-8,
                 learning_rate: float = 1e-2,
                 kl_step: int = 200,
                 u0: np.ndarray = None,
                 init_sigma: float = 1.0,
                 low_range: np.ndarray = None,
                 high_range: np.ndarray = None,
                 batch_size: int = 4, time_step: int = 25,
                 is_stochastic: bool = True,
                 states: np.ndarray = None,
                 threashold: float = 4e-1
                 ):
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
        self.N = N  # Number of fitted iLQG controlllers.
        self.M = M  # Number of fitted MPC controllers per iLQG c.
        self.T = env.steps  # Number of steps done in a simulation.

        self.n_x = env.observation_space.shape[0]  # State dimention.
        self.n_u = env.action_space.shape[0]       # Action dimention.

        # Transformation for control arguments
        self.t_x = t_x  # x_t -> o_t
        self.t_u = t_u  # u_t -> a_t
        self.inv_t_u = inv_t_u  # a_t -> u_t
        self.inv_t_x = inv_t_x  # o_t -> x_t

        # Instances for simulation.
        self.env = env
        if isinstance(low_range, np.ndarray) or isinstance(low_range, list):
            self.low_range = low_range
        else:
            self.low_range = self.env.observation_space.low

        if isinstance(high_range, np.ndarray) or isinstance(high_range, list):
            self.high_range = high_range
        else:
            self.high_range = self.env.observation_space.high

        # Lagrange's multipliers
        self.lamb = lamb * np.ones((N, self.T, self.n_u))   # contraint weight.
        # learning rate for lamb.
        self.alpha_lamb = alpha_lamb
        self.nu = nu * np.ones((N, self.T))               # KL-div weight.
        # old traj weight for iLQG.
        self.eta = eta * np.ones(N)
        self.kl_step = kl_step
        self.is_stochastic = is_stochastic

        # ilQG instances.
        if not callable(cost_terminal):
            def cost_terminal(x, i): return cost(x, np.zeros(self.n_u), i)

        self.dynamics_kwargs = dynamics_kwargs
        u0 = np.zeros(self.n_u) if u0 is None else u0
        self.cost_kwargs = dict(cost=cost,
                                l_terminal=cost_terminal,
                                n_x=self.n_x,
                                n_u=self.n_u,
                                eta=eta,
                                nu=nu * np.ones(self.T),
                                lamb=lamb * np.ones(self.n_u),
                                T=self.T,
                                u0=u0)

        self.policy = policy.to(device)
        self.policy_sigma = init_sigma * np.identity(self.n_u)
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate)

        # Buffer's instances.
        self.batch_size = batch_size
        self.buffer = iLQR_Rollouts(N, M, self.T,
                                    self.n_u,
                                    self.policy.state_dim,
                                    time_step=time_step)

        self._regions, self.x0 = self._get_regions(states, threashold)

    def _get_regions(self, states: np.ndarray = None, threashold: float = 4e-1) -> Union[List, np.ndarray]:
        regions = list()
        region = UniformSet(self.low_range, self.high_range)
        regions += [region]
        centroids = list()
        centroids += [region.centroid]
        if isinstance(states, np.ndarray):
            mask = confidence_region(states[:, :, -1], c=threashold, ord='inf')
            self.N = states.shape[0] + 1
            for i in range(states.shape[0]):
                selected_states = states[0, 0, 0] > 0.0
                filtered_states = states[i][mask[i]]
                filtered_start_states = filtered_states[:, 0]
                coordinates = filtered_start_states[:, selected_states]
                region = ConvexSet(coordinates, selected_states, self.n_x)
                centroids += [region.centroid]
                regions += [region]
        else:
            for i in range(self.N - 1):
                centroid = self.env.observation_space.sample()
                regions += [UniformSet(self.low_range,
                                       self.high_range, centroid)]
                centroids += [centroid]

        return regions, np.array(centroids)

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
        nominal_xs = np.expand_dims(nominal_xs, axis=1)
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
        return C_new

    def policy_loss(self, states, actions, lamb, nu, C):
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
        if (len(C.shape) == 4) and (len(actions.shape) == 4):
            C = np.expand_dims(C, axis=1)
        if (len(lamb.shape) == 3) and (len(actions.shape) == 4):
            lamb = np.expand_dims(lamb, axis=1)
        if (len(nu.shape) == 2) and (len(actions.shape) == 4):
            nu = np.expand_dims(nu, axis=1)

        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states)
        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions)
        if not isinstance(C, torch.Tensor):
            C = torch.FloatTensor(C)
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.FloatTensor(lamb)
        if not isinstance(nu, torch.Tensor):
            nu = torch.FloatTensor(nu)

        states = states.to(device)
        actions = actions.to(device)
        C = C.to(device)
        lamb = lamb.to(device)
        nu = nu.to(device)

        # 1. Cálculo de \mu^{\pi}(x)
        policy_actions = self.inv_t_u(self.policy(states))

        # 2. Cálculo de perdida
        loss = 0.0
        # 2.1 Cálculo de la distancia Mahalanobis
        loss += _batch_mahalanobis(C, policy_actions - actions)

        kld = loss
        # 2.2 Multiplicación escalar de
        if len(loss.shape) == 3:
            loss = torch.einsum('NMT, NMT -> NMT', nu, loss)
        else:
            loss = nu * loss

        # 3. Cálculo de terminos de regularización
        if len(lamb.shape) == 4:
            loss += 2 * torch.einsum('NMTu, NMTu -> NMT', lamb, policy_actions)
            loss = torch.sum(loss, dim=-1)
        else:
            loss += 2 * torch.einsum('Nu, Nu -> N', lamb, policy_actions)

        # Cálculo de perdida promedio
        loss = 0.5 * torch.mean(loss)
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
        eta = np.empty(self.N)
        div = np.empty(self.N)
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
            eta[i] = file['eta']
            div[i] = file['div']
        nominal_xs = nominal_xs[:, :-1]
        xs = xs[:, :, :-1]
        return K, k, C, nominal_xs, nominal_us, xs, us, alphas, eta, div

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
        mean_div = np.mean(div, axis=-1)  # (N,)
        std_div = np.std(div, axis=-1)
        mask = np.apply_along_axis(lambda x: np.greater(x, mean_div), 0, div)
        nu[mask] *= 2.0
        mask = np.apply_along_axis(lambda x: np.less(x, 2 * std_div), 0, div)
        nu[mask] /= 2.0
        return nu

    def init_x0(self, inlude_origin=True):
        if self.N > 1:
            self.x0 = np.array([self.env.observation_space.sample()
                               for _ in range(self.N)])
            if inlude_origin:
                self.x0[0] = np.zeros(self.n_x)
        else:
            self.x0 = self.env.observation_space.sample()

    def _random_x0(self, x0, n):
        size = (n, self.n_x)
        x = np.random.uniform(self.low_range, self.high_range, size)
        return x + x0

    def fit_policy(self, dataloader: DataLoader):
        self.policy.train()
        n = 0
        avg_loss = 0.0
        for _, batch in enumerate(dataloader, 0):
            for data in batch:
                loss = self.policy_loss(*data)[0]
                # 2.1 Optimization step
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                loss = loss.detach().cpu().item()
                n += 1
                avg_loss += loss
        return avg_loss / n

    def update_policy(self, path, constrained_actions=False,
                      shuffle_batches=False, policy_updates=2):
        pathlib.Path(path + 'buffer/').mkdir(parents=True, exist_ok=True)
        path = path + 'buffer/'
        if constrained_actions:
            low_constrain = self.env.action_space.low
            high_constain = self.env.action_space.high
        else:
            low_constrain = None
            high_constain = None

        # 1. Control iLQR fitting
        self.policy.eval()
        if self.N > 1:
            processes = list()
            for i in range(self.N):
                # self._random_x0(self.x0[i], self.M)
                x0_samples = self._regions[i].sample(self.M)
                cost_kwargs = deepcopy(self.cost_kwargs)
                cost_kwargs['lamb'] = self.lamb[i]
                cost_kwargs['nu'] = self.nu[i]
                cost_kwargs['eta'] = self.eta[i]
                p = mp.Process(target=fit_ilqg,
                               args=(self.x0[i],
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
                                     x0_samples,
                                     low_constrain,
                                     high_constain,
                                     self.is_stochastic
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
            x0_samples = self._regions[0].sample(self.M)
            fit_ilqg(self.x0, self.kl_step, self.policy, cost_kwargs,
                     self.dynamics_kwargs, i, self.T, self.M, path,
                     self.t_x, self.inv_t_u, self.policy._sigma,
                     x0_samples, low_constrain, high_constain,
                     self.is_stochastic
                     )

        # 1.1 Loading fitted parameters and simulations
        (K, k, C, nominal_xs, nominal_us,
         xs, us, alphas, eta, control_div) = self._load_fitted_lqg(path)
        self.eta = eta
        # 2. Policy fitting

        # 2.1 Calculate mean action
        us_mean = self.mean_control(xs, nominal_xs, nominal_us, K, k, alphas)
        if callable(self.t_x):
            xs = np.apply_along_axis(self.t_x, -1, xs)  # x_t -> o_t

        # 2.2 Create Dataset and Dataloader instances
        self.buffer.update_rollouts(xs, us_mean, self.lamb, self.nu, C)

        # 2.3 Mean policy fitting (neural network)
        dataloader = DataLoader(
            self.buffer,
            batch_size=self.batch_size,
            shuffle=shuffle_batches)
        for _ in range(policy_updates):
            loss = self.fit_policy(dataloader)

        # 2.4 Update policy covariance matrix
        self.policy._sigma = self.cov_policy(C)

        # 3. Update Langrange's multipliers
        self.policy.eval()
        us_policy = self.policy.get_action(xs)
        if callable(self.inv_t_u):  # a_t -> u_t
            us_policy = np.apply_along_axis(self.inv_t_u, -1, us_policy)

        # 3.1 Update lamb
        self.lamb = self._update_lamb(us_policy, us)

        # 3.2 Update nu
        loss, policy_div = self.policy_loss(xs, us_mean, self.lamb, self.nu, C)
        policy_div = np.sum(
            policy_div.detach().cpu().numpy(), axis=-1)  # (NM,)
        loss = loss.detach().cpu().item()
        mean_div = np.mean(policy_div, axis=1)  # (N, T)
        self.nu = self._update_nu(mean_div)

        return loss, policy_div, control_div


def fit_ilqg(x0, kl_step, policy, cost_kwargs, dynamics_kwargs, i, T, M,
             path='', t_x=None, inv_t_u=None, policy_sigma=None,
             x0_samples=None, low_constrain=None, high_constrain=None,
             is_stochastic=False):
    '''
    Argumentos
    ----------
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
    control = OfflineController(dynamics, cost, T)
    # Actualización de parametros de control para costo
    file_name = f'control_{i}.npz'
    if exists(path + file_name):
        # file_name = file_name if exists(file_name) else 'control.npz'
        control.load(path, file_name)
    else:
        expert = iLQG(dynamics, cost, T, is_stochastic=False)
        expert.load('models/', f'ilqr_control_{int(T)}.npz')
        us_init = expert.rollout(x0)[1]
        cost.nu = np.zeros(control.N)
        cost.update_control(control=expert)
        _ = control.fit_control(x0, us_init=us_init)

    # También puede actualizar eta
    cost.update_control(control)
    is_stochastic = control.is_stochastic
    control.is_stochastic = False
    xs, us = control.rollout(x0)
    control.is_stochastic = is_stochastic
    # control.fit_control(xs[0], us_init)
    control.x0, control.us_init = x0, us
    kl_div = control.optimize(
        kl_step, min_eta=cost.eta, max_eta=eval(PARAMS_OFFLINE['max_eta']))[-1]
    print(f'- eta={cost.eta}, kl_div={kl_div}')
    control.save(path=path, file_name=f'control_{i}.npz')

    states = np.empty((M, T + 1, n_x))
    actions = np.empty((M, T, n_u))
    control.is_stochastic = is_stochastic
    if not isinstance(x0_samples, np.ndarray):
        x0_samples = multivariate_normal.rvs(
            mean=x0, cov=0.01 * np.identity(n_x), size=M)
    for r in range(M):
        xs, us = control.rollout(x0_samples[r],
                                 low_constrain=low_constrain,
                                 high_constrain=high_constrain)
        states[r] = xs
        actions[r] = us

    np.savez(path + f'rollouts_{i}.npz',
             xs=states,
             us=actions
             )
    # Ajuste MPC
