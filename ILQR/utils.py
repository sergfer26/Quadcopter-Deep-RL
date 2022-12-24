import numpy as np
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


class DiffCostNN(FiniteDiffCost):

    def __init__(self, cost_net, state_size, action_size, t_x=None, t_u=None, x_eps=None, u_eps=None,
                 u_bound=None, q1=q1, q2=q2):
        self.cost_net = cost_net

        def cost(x, u, i):
            cost = 0.0
            if isinstance(u_bound, np.ndarray):
                cost += q1 * np.exp(q2 * (u ** 2 - u_bound ** 2)).sum()
            cost += cost_net.to_float(x, u, t_x=t_x, t_u=t_u)
            return cost

        def cost_terminal(x, i):
            u = np.zeros(action_size)
            return cost_net.to_float(x, u, t_x=t_x, t_u=t_u)
        super().__init__(cost, cost_terminal, state_size, action_size, x_eps, u_eps)


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
            c /= (self.eta + nu[i])
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
        self.nu = nu * np.ones(N)
        self._lambda = _lambda if _lambda is not None else np.ones(action_size)
        # Parameters of the nonlinear policy
        self.cov = np.identity(action_size)
        self.mean = lambda x: np.ones(
            action_size) if not callable(mean) else mean

        def _cost(x, u, i):
            c = 0.0
            c -= u.T@self._lambda
            c -= nu[i] * self.nu[i] * \
                m_n.logpdf(x=u, mean=self.mean(x), cov=self.cov)
            c -= m_n.logpdf(x=u, mean=self._K[i]
                            @ x + self._k[i], cov=self._C[i])

        super().__init__(_cost, lambda x, i: 0, state_size, action_size, x_eps, u_eps)

    def update_control(self, control):
        self._K = control._K
        self._k = control._k
        self._C = control._C
