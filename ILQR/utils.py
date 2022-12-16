import numpy as np
from scipy.integrate import odeint
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.cost import FiniteDiffCost
from .params import PARAMS_iLQR

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
