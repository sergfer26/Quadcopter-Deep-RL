import numpy as np
import random
import numpy as np
from collections import deque
from scipy.integrate import odeint
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.cost import FiniteDiffCost
from .params import PARAMS_iLQR
from scipy.stats import multivariate_normal as m_n

q1 = PARAMS_iLQR['q1']
q2 = PARAMS_iLQR['q2']


class ContinuousDynamics(FiniteDiffDynamics):

    def __init__(self, f, state_size, action_size, dt=0.1, x_eps=None, u_eps=None, u0=None, method='lsoda'):
        # def f_d(x, u, i): return x + f(x, u)*dt
        if not isinstance(u0, np.ndarray):
            u0 = np.zeros(action_size)

        if method == 'euler':
            def f_d(x, u, i):
                w1, w2, w3, w4 = u + u0
                return x + f(x, i*dt, w1, w2, w3, w4)*dt
        elif method == 'lsoda':
            def f_d(x, u, i):
                w1, w2, w3, w4 = u + u0
                t = [i * dt, (i+1)*dt]
                return odeint(f, x, t, args=(w1, w2, w3, w4))[1]

        super().__init__(f_d, state_size, action_size, x_eps, u_eps)


class FiniteDiffCostBounded(FiniteDiffCost):

    def __init__(self, cost, l_terminal, state_size, action_size, x_eps=None,
                 u_eps=None, u_bound=None, q1=q1, q2=q2):
        def l_bounded(x, u, i):
            total = cost(x, u, i)
            if isinstance(u_bound, np.ndarray):
                total += q1 * np.exp(q2 * (u ** 2 - u_bound ** 2)).sum()
            return total

        super().__init__(l_bounded, l_terminal, state_size, action_size, x_eps, u_eps)


class OfflineCost(FiniteDiffCost):

    def __init__(self, l, l_terminal, state_size, action_size,
                 x_eps=None, u_eps=None,
                 eta=0.1, nu=0.001, _lambda=None,
                 N=10, mean=None
                 ):

        # Steps of the trajectory
        self.N = N
        # Parameters of the old control
        self._k = np.zeros((N, action_size))
        self._K = np.zeros((N, action_size, state_size))
        self._C = np.stack([np.identity(action_size)
                            for _ in range(self.N)])
        # Lagrange's multipliers
        self.eta = eta
        self.nu = nu * np.ones(N)
        self._lambda = _lambda if _lambda is not None else np.ones(action_size)
        # Parameters of the nonlinear policy
        self.cov = np.identity(action_size)
        self.mean = lambda x: np.ones(
            action_size) if not callable(mean) else mean

        def _cost(x, u, i):
            c = 0.0
            c += l(x, u, i) - u.T@self._lambda
            c -= self.nu[i] * m_n.logpdf(x=u, mean=self.mean(x), cov=self.cov)
            c /= (self.eta + self.nu[i])
            c -= eta * m_n.logpdf(x=u, mean=self._K[i]
                                  @ x + self._k[i], cov=self._C[i]) / (self.eta + self.nu[i])
            return c

        super().__init__(_cost, l_terminal, state_size, action_size, x_eps, u_eps)

    def update_control(self, control):
        self._K = control._K
        self._k = control._k
        self._C = control._C


class MPCCost(FiniteDiffCost):

    def __init__(self, state_size, action_size,
                 x_eps=None, u_eps=None,
                 nu=0.001, _lambda=None,
                 N=10, mean_policy=None,
                 ):
        # Steps of the trajectory
        self.N = N
        # Parameters of the old control
        self.old_x = np.empty(state_size)
        self.old_u = np.empty(action_size)

        # Lagrange's multipliers
        self.nu = nu * np.ones(N)
        self._lambda = _lambda if _lambda is not None else np.ones(action_size)
        # Parameters of the nonlinear policy
        self.cov = np.identity(action_size)
        self.mean_policy = lambda x: np.ones(
            action_size) if not callable(mean_policy) else mean_policy
        self.cov_dynamics = np.stack(
            [np.identity(state_size) for _ in range(N)])
        self.mean_dynamics = np.zeros((N, state_size))

        super().__init__(self._cost, lambda x, i: 0, state_size, action_size, x_eps, u_eps)

    def update_dynamics(self, control):
        self.mean_dynamics = control._mean
        self.cov_dynamics = control._sigma

    def _cost(self, x, u, i):
        c = 0.0
        c -= u.T@self._lambda
        # log policy distribution
        c -= self.nu[i] * \
            m_n.logpdf(x=u, mean=self.mean_policy(x), cov=self.cov)
        # log dynamics distribution
        try:
            c -= m_n.logpdf(x=x,
                            mean=self.mean_dynamics[i],
                            cov=self.cov_dynamics[i])
        except:
            breakpoint()
        return c


class Memory:

    def __init__(self, max_size, action_dim, state_dim, T):
        self.max_size = max_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.T = T
        self.buffer = deque(maxlen=max_size)

    def push(self, trajectory=None, **kwargs):
        '''
        Método para agregar trayectorias al buffer de la memoria.

        Argumentos
        ----------
        trajectory (ocional): list
            La lista con las tuplas que representan la experiencia a lo largo
            de un episodio. Cada tupla debe tener la siguiente estructura:
            (state, action, new_state, done). `len(trajectory) -> self.T`.
        states : `np.ndarray`
            Es un arreglo que representa los estados de una o `m` simulaciones.
            `states.shape -> (self.T, self.state_dim)`
            o `states.shape -> (m, self.T, self.state_dim)`
        actions : `np.ndarray`
            Es un arrreglo que representa las acciones de una o `m`
            simulaciones.
            `actions.shape -> (self.T, self.action_dim)`
            o `states.shape -> (m, self.T, self.action_dim)`
        next_states : `np.ndarray`
            Es un arreglo que representa los nuevos estados de una o `m`
            simulaciones.
            `next_states.shape -> (self.T, self.state_dim)`
            o `next_states.shape -> (m, self.T, self.state_dim)`
        dones : `np.array`
            Es un arreglo que representa los booleanos de termino de una
            simulación o `m` simulaciones.
            `dones.shape -> self.T` o `next_states.shape -> (m, self.T)`
        '''
        if isinstance(trajectory, list):
            states, actions, next_states, dones = self._preprocess_traj(
                trajectory)
        elif len(kwargs) > 0:
            states = kwargs['states']
            actions = kwargs['actions']
            next_states = kwargs['next_states']
            dones = kwargs['dones']
        if len(states.shape) == 2:
            x = [states, actions, next_states, dones]
            self.buffer.append(x)
        elif len(states.shape) == 3:
            for i in range(states.shape[0]):
                x = [states[i, :, :], actions[i, :, :],
                     next_states[i, :, :], dones[i, :]]
                self.buffer.append(x)
        else:
            print('No pudo procesar la información')

    def sample(self, sample_size, t_x=None, t_u=None):
        '''
        Muestrea trayectorias de simulaciones.

        Argumentos
        ----------
        sample_size : int
            Representa el tamaño de muestra.

        Retornos
        --------
        states : `np.ndarray`
            Arreglo que representa los estados de `sample_size` simulaciones.
            `states.shape -> (sample_size, self.T, self.state_dim)`.
        actions : `np.ndarray`
            Arreglo que representa las acciones de `sample_size` simulaciones.
            `states.shape -> (sample_size, self.T, self.action_dim)`.
        new_states : `np.ndarray`
            Arreglo que representa los nuevos estados de `sample_size`
            simulaciones.
            `states.shape -> (sample_size, self.T, self.state_dim)`.
        dones : `np.ndarray`
            Arreglo que representa los booleanos de terminado de
            `sample_size` simulaciones.
            `dones.shape -> (sample_size, self.T)`.
        '''
        samples = random.sample(self.buffer, sample_size)
        states = np.empty((sample_size, self.T, self.state_dim))
        actions = np.empty((sample_size, self.T, self.action_dim))
        new_states = np.empty((sample_size, self.T, self.state_dim))
        dones = np.empty((sample_size, self.T))
        for i in range(sample_size):
            states[i, :, :] = samples[i][0]
            actions[i, :, :] = samples[i][1]
            new_states[i, :, :] = samples[i][2]
            dones[i, :] = samples[i][3]
        if callable(t_x):
            states = np.apply_along_axis(t_x, -1, states)
            new_states = np.apply_along_axis(t_x, -1, new_states)
        if callable(t_u):
            actions = np.apply_along_axis(t_u, -1, actions)
        return states, actions, new_states, dones

    def _preprocess_traj(self, trajectory):
        states = np.zeros((self.T, self.state_dim))
        actions = np.zeros((self.T, self.action_dim))
        next_states = np.zeros((self.T, self.state_dim))
        dones = np.zeros((self.T, 1))
        for i, experience in enumerate(trajectory):
            state, action, next_state, done = experience
            states[i, :] = state
            actions[i, :] = action
            next_states[i, :] = next_state
            dones[i, :] = done
        return states, actions, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
