import numpy as np
import warnings
from os.path import exists
from ilqr.controller import iLQR

from scipy.integrate import odeint
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.cost import FiniteDiffCost
from scipy.stats import multivariate_normal as normal


def mvn_kl_div(mu1, mu2, sigma1, sigma2):
    '''
    Kullback-Leiber divergence for a multivariate normal
    distribution X1 from another mvn X2.
    https://statproofbook.github.io/P/mvn-kl.html

    Arguments
    ---------
    mu1 : `np.ndarray`
        Mean of X1 as numerical array with dimension (n,).
    mu2 _ `np.ndarray`
        Mean of X2 as numerical array with dimension (n,).
    sigma1: `np.ndarray`
        Covariance matrix of X1 as numerical array with dimensions (n, n).
    sigma2 : `np.ndarray`
        Covariance matrix of X2 as numerical array with dimensions (n, n).

    Return
    ------
    kl_div : `float`
        divergence measure.
    '''
    n = mu1.shape[0]
    inv_sigma2 = np.linalg.inv(sigma2)
    div = 0.0
    div += (mu2 - mu1).T @ inv_sigma2 @ (mu2 - mu1)
    div += np.trace(inv_sigma2 @ sigma1)
    div -= np.log(np.linalg.det(sigma1)/np.linalg.det(sigma2))
    div -= n
    return 0.5 * div


class ContinuousDynamics(FiniteDiffDynamics):

    def __init__(self, f, n_x, n_u, dt=0.1, x_eps=None, u_eps=None, u0=None, method='lsoda'):
        # def f_d(x, u, i): return x + f(x, u)*dt
        if not isinstance(u0, np.ndarray):
            u0 = np.zeros(n_u)

        if method == 'euler':
            def f_d(x, u, i):
                w1, w2, w3, w4 = u + u0
                return x + f(x, i*dt, w1, w2, w3, w4)*dt
        elif method == 'lsoda':
            def f_d(x, u, i):
                w1, w2, w3, w4 = u + u0
                t = [i * dt, (i+1)*dt]
                return odeint(f, x, t, args=(w1, w2, w3, w4))[1]
        super().__init__(f_d, n_x, n_u, x_eps, u_eps)


class OfflineCost(FiniteDiffCost):

    def __init__(self, cost, l_terminal, n_x, n_u,
                 x_eps=None, u_eps=None,
                 eta=0.1, nu=None, lamb=None,
                 T=10, policy_mean=None, policy_cov=None
                 ):
        '''Constructs an Offline cost for Guided policy search.

        Args:
            l: Instantaneous cost function to approximate.
                Signature: (x, u, i) -> scalar.
            l_terminal: Terminal cost function to approximate.
                Signature: (x, i) -> scalar.
            n_x: State size.
            n_u: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            eta : Langrange's multiplier, old policy weight.
                Default: 0.1.
            nu : Langrange's multiplier, neural network weight.
                Default: 0.001.
            lambd : Langrange's multiplier, contraint weight.
                Default: None (np.array).
        '''
        # Steps of the trajectory
        self.T = T
        # Parameters of the old control
        self._k = np.zeros((T, n_u))
        self._K = np.zeros((T, n_u, n_u))
        self._C = np.stack([np.identity(n_u)
                            for _ in range(T)])
        # Lagrange's multipliers
        self.eta = eta
        self.nu = 0.001 * np.ones(T) if nu is None else nu
        self.lamb = 0.001 * np.ones((T, n_u)) if lamb is None else lamb
        # Parameters of the nonlinear policy
        self.policy_cov = np.identity(n_u) if not callable(
            policy_cov) else policy_cov
        self.policy_mean = lambda x: np.zeros(
            n_u) if not callable(policy_mean) else policy_mean

        self.cost = cost  # l_bounded

        def _cost(x, u, i):
            C = self._C[i] + 5e-1 * np.identity(u.shape[-1])
            c = 0.0
            c += self.cost(x, u, i) - u.T@self.lamb[i]
            c -= self.nu[i] * \
                normal.logpdf(x=u, mean=self.policy_mean(x),
                              cov=self.policy_cov)
            c /= (self.eta + self.nu[i])
            mean_control = self._control(x, i)
            c -= self.eta * normal.logpdf(x=u, mean=mean_control,
                                          cov=C) / (self.eta + self.nu[i])
            return c

        super().__init__(_cost, l_terminal, n_x, n_u, x_eps, u_eps)

    def _control(self, x, i):
        us = self._us[i]
        xs = self._xs[i]
        return us + self._alpha * self._k[i] + self._K[i] @ (x - xs)

    def update_control(self, control: iLQR = None, file_path: str = None):
        if isinstance(control, iLQR):
            self._K = control._K.copy()
            self._k = control._k.copy()
            self._C = control._C.copy()
            self._xs = control._nominal_xs.copy()
            self._us = control._nominal_us.copy()
            self._alpha = 0.0 + control.alpha
        elif exists(file_path):
            file = np.load(file_path)
            self._K = file['K']
            self._k = file['k']
            self._C = file['C']
            self._xs = file['xs']
            self._us = file['us']
            self._alpha = file['alpha']
            if 'eta' in file.files:
                self.eta = file['eta']
        else:
            warnings.warn("No path nor control was provided")
        # self.control_updated = True

    def control_parameters(self):
        return dict(
            xs=self._xs,
            us=self._us,
            k=self._k,
            K=self._K,
            C=self._C,
            alpha=self._alpha
        )


class OnlineCost(FiniteDiffCost):

    def __init__(self, n_x, n_u,
                 control,
                 x_eps=None, u_eps=None,
                 nu=0.001, lamb=None,
                 policy_mean=None
                 ):
        '''Constructs an Online cost for Guided policy search.

        Args:
            n_x: State size.
            n_u: Action size.
            control: ilqr.iLQR instance controller.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            eta : Langrange's multiplier, old policy weight.
                Default: 0.1.
            nu : Langrange's multiplier, neural network weight.
                Default: 0.001.
            lamb : Langrange's multiplier, contraint weight.
                Default: None (np.array).
        '''
        self.control = control
        self.n_x = n_x
        self.n_u = n_u

        # Steps of the trajectory
        self.N = control.N

        # Lagrange's multipliers
        self.nu = nu * np.ones(self.N)
        self.lamb = lamb if lamb is not None else np.ones(n_u)
        # Parameters of the nonlinear policy
        self.policy_cov = np.identity(n_u)
        self.policy_mean = lambda x: np.ones(
            n_u) if not callable(policy_mean) else policy_mean
        self.cov_dynamics = np.stack(
            [np.identity(n_x) for _ in range(self.N)])
        self.mean_dynamics = np.zeros((self.N, n_x))
        self._F = 1e-3 * np.identity(self.n_x)

        super().__init__(self._cost, lambda x, i: 0, n_x, n_u, x_eps, u_eps)

    def _mean_dynamics(self, x, u, i, mu=None):
        r'''
        Cálcula la media de la dinamica del sistema
        $$
        \mu_{t^{'}} = \mathbb{E}[X \sim p(x_t^{'}|x_{t})]
        \forall t^{'} \in [t+1, t + H]
        $$
        '''
        if not isinstance(mu, np.ndarray):
            mu = np.zeros(self.n_x)
        f_x = self.control.dynamics.f_x(x, u, i)
        f_u = self.control.dynamics.f_u(x, u, i)
        u_hat = self.control._nominal_us[i]
        x_hat = self.control._nominal_xs[i]
        k = self.control._k[i]
        K = self.control._K[i]

        f_xu = np.hstack([f_x, f_u])
        new_u = u_hat + self.control.alpha * k + K @ (mu - x_hat)
        return f_xu @ np.hstack([mu, new_u])

    def _cov_dynamics(self, x, u, i, sigma=None):
        if not isinstance(sigma, np.ndarray):
            sigma = np.zeros_like(self._F)
        f_x = self.control.dynamics.f_x(x, u, i)
        f_u = self.control.dynamics.f_u(x, u, i)
        f_xu = np.hstack([f_x, f_u])
        K = self.control._K[i]
        C = self.control._C[i]
        a1 = np.hstack([sigma, sigma @ K.T])
        try:
            a2 = np.hstack([K @ sigma, C + K @ sigma @ K.T])
        except:
            breakpoint()
        sigma = f_xu @ np.vstack([a1, a2]) @ f_xu.T + self._F
        return sigma

    def _dist_dynamics(self, xs, us):
        mu = xs[0]
        sigma = self._F
        N = us.shape[0]
        for i in range(N):
            mu = self._mean_dynamics(xs[i], us[i], i=i, mu=mu)
            sigma = self._cov_dynamics(xs[i], us[i], i=i, sigma=sigma)
            self.mean_dynamics[i] = mu
            self.cov_dynamics[i] = np.round(sigma, 5)

    def update_control(self, control):
        self.control = control

    def _cost(self, x, u, i):
        c = 0.0
        c -= u.T@self.lamb
        # log policy distribution
        c -= self.nu[i] * \
            normal.logpdf(x=u, mean=self.policy_mean(x), cov=self.policy_cov)
        # log dynamics distribution
        try:
            c -= normal.logpdf(x=x,
                               mean=self.mean_dynamics[i],
                               cov=self.cov_dynamics[i])
        except:
            breakpoint()
        return c


def rollout(agent, env, flag=False, state_init=None):
    '''
    Simulación de interacción entorno-agente

    Argumentos
    ----------
    agent : `(DDPG.DDPGAgent, Linear.Agent, GPS.iLQRAgent)`
        Instancia que representa al agente que toma acciones 
        en la simulación.
    env : `gym.Env`
        Entorno de simualción de gym.
    flag : bool
        ...
    state_init : `np.ndarray`
        Arreglo que representa el estado inicial de la simulación.

    Retornos
    --------
    states : `np.ndarray`
        Trayectoria de estados en la simulación con dimensión (env.steps, n_x).
    acciones : `np.ndarray`
        Trayectoria de acciones en la simulación con dimensión 
        (env.steps -1, n_u).
    scores : `np.ndarray`
        Trayectoria de puntajes (incluye reward) en la simulación con
        dimensión (env.steps -1, ?).
    '''
    # t = env.time
    env.flag = flag
    state = env.reset()
    if hasattr(agent, 'reset') and callable(getattr(agent, 'reset')):
        agent.reset()
    if isinstance(state_init, np.ndarray):
        env.state = state_init
        state = state_init
    states = np.zeros((env.steps, env.observation_space.shape[0]))
    actions = np.zeros((env.steps - 1, env.action_space.shape[0]))
    scores = np.zeros((env.steps - 1, 2))  # r_t, Cr_t
    states[0, :] = state
    episode_reward = 0
    i = 0
    while True:
        action = agent.get_action(state)
        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        states[i + 1, :] = state
        if isinstance(info, dict) and ('real_action' in info.keys()):
            action = info['real_action']  # env.action(action)
        actions[i, :] = action
        scores[i, :] = np.array([reward, episode_reward])
        state = new_state
        if done:
            break
        i += 1
    return states, actions, scores
