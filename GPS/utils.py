import numpy as np
import warnings
import torch
from os.path import exists
from ilqr.controller import iLQR

from scipy.integrate import odeint
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.cost import FiniteDiffCost
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
from torch.utils.data import Dataset
from .params import PARAMS_LQG, PARAMS_ONLINE
from .policy import Policy
# from scipy.linalg import issymmetric

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class iLQR_Rollouts(Dataset):

    def __init__(self, N, M, T, n_u, n_x,
                 is_sequential=True,
                 time_step=25, shuffle=True):
        self.N = N
        self.M = M
        self.T = T
        self.n_u = n_u
        self.n_x = n_x
        self.is_sequential = is_sequential
        self.time_step = time_step
        self.shuffle = shuffle

    def __len__(self):
        if self.is_sequential:
            length = self.M
        else:
            length = self.N * self.T * self.M
        return length

    def __getitem__(self, index):
        '''
        Retornos
        --------
        states, actions, lamb, nu, C
        '''
        if self.is_sequential:
            steps = self.T // self.time_step
            out = [(self.states[:, index, t * self.time_step:
                                (t+1) * self.time_step],
                    self.actions[:, index,  t * self.time_step:
                                 (t+1) * self.time_step],
                    self.lamb[:, index, t * self.time_step:
                              (t+1) * self.time_step],
                    self.nu[:, index, t * self.time_step:
                            (t+1) * self.time_step],
                    self.C[:, index, t * self.time_step:
                           (t+1) * self.time_step]) for t in range(steps)]
        else:
            out = (self.states[index], self.actions[index], self.lamb[index],
                   self.nu[index], self.C[index])
        return out

    def update_rollouts(self, states, actions, lamb, nu, C):
        # iLQR parameters
        C = np.repeat(np.expand_dims(C, axis=1), self.M, axis=1)
        lamb = np.repeat(np.expand_dims(lamb, axis=1), self.M, axis=1)
        nu = np.repeat(np.expand_dims(nu, axis=1), self.M, axis=1)

        if self.shuffle:
            indices = np.random.permutation(states.shape[2])
            states = states[:, :, indices]
            actions = actions[:, :, indices]
            lamb = lamb[:, :, indices]
            nu = nu[:, :, indices]
            C = C[:, :, indices]
        if not self.is_sequential:
            C = C.reshape(self.N * self.M * self.T, self.n_u, self.n_u)
            lamb = lamb.reshape(self.N * self.M * self.T, self.n_u)
            nu = nu.reshape(self.N * self.M * self.T)

            states = states.reshape(self.N * self.M * self.T, self.n_x)
            actions = actions.reshape(self.N * self.M * self.T, self.n_u)

        self.C = torch.FloatTensor(C)
        self.lamb = torch.FloatTensor(lamb)
        self.nu = torch.FloatTensor(nu)

        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)


class ContinuousDynamics(FiniteDiffDynamics):

    def __init__(self, f, n_x, n_u,
                 dt=0.1, x_eps=None, u_eps=None, u0=None):
        # def f_d(x, u, i): return x + f(x, u)*dt
        if not isinstance(u0, np.ndarray):
            u0 = np.zeros(n_u)

        def f_d(x, u, i):
            w1, w2, w3, w4 = u + u0
            t = [i * dt, (i+1)*dt]
            return odeint(f, x, t, args=(w1, w2, w3, w4))[1]
        super().__init__(f_d, n_x, n_u, x_eps, u_eps)


class OfflineCost(FiniteDiffCost):

    def __init__(self, l, l_terminal, state_size, action_size,
                 x_eps=None, u_eps=None,
                 eta=0.1, nu=None, lamb=None,
                 T=10, policy_mean=None, policy_cov=None,
                 known_dynamics=True,
                 u0=None):
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
            nu : Langrange's multiplier for neural network.
                Default: 0.001.
            lambd : Langrange's multiplier, contraint weight.
                Default: None (np.array).
        '''
        # Steps of the trajectory
        self.T = T
        # Parameters of the old control
        self._k = np.zeros((T, action_size))
        self._K = np.zeros((T, action_size, state_size))
        self._C = np.stack([np.identity(action_size)
                            for _ in range(T)])
        # Lagrange's multipliers
        self.eta = eta
        self.nu = 0.001 * np.ones(T) if nu is None else nu
        self.lamb = 0.001 * np.ones((T, action_size)) if lamb is None else lamb
        # Parameters of the nonlinear policy
        self.policy_cov = np.identity(action_size) if not callable(
            policy_cov) else policy_cov
        self.policy_mean = None  # lambda x: np.zeros(
        # n_u) if not callable(policy_mean) else policy_mean

        self.cost = l  # l_bounded
        self.known_dynamics = known_dynamics
        self.u0 = np.zeros(action_size) if u0 is None else u0
        super().__init__(self._cost, l_terminal, state_size,
                         action_size, x_eps, u_eps)

    def _cost(self, x, u, i):
        C = self._C[i]
        c = 0.0
        c += self.cost(x, u, i) - (self.u0 + u).T@self.lamb[i]
        if callable(self.policy_mean):
            c -= self.nu[i] * logpdf(x=u, mean=self.policy_mean(x),
                                     cov=self.policy_cov)
        c /= (self.eta + self.nu[i])
        c -= self.eta * logpdf(x=u, mean=self._control(x, i),
                               cov=C) / (self.eta + self.nu[i])
        return c

    def _control(self, x, i):
        us = self._us[i]
        xs = self._xs[i]
        return us + self._alpha * self._k[i] + self._K[i] @ (x - xs)

    def update_control(self, control: iLQR = None, file_path: str = None):
        if isinstance(control, iLQR):
            self._K = control._K.copy()
            self._k = control._k.copy()
            self._xs = control._nominal_xs.copy()
            self._us = control._nominal_us.copy()
            self._alpha = 0.0 + control.alpha
            C = control._C.copy()
        elif exists(file_path):
            file = np.load(file_path)
            self._K = file['K']
            self._k = file['k']
            self._xs = file['xs']
            self._us = file['us']
            self._alpha = file['alpha']
            C = file['C']
            if 'eta' in file.files:
                self.eta = file['eta']
        else:
            warnings.warn("No path nor control was provided")
        self._C = np.array([nearestPD(C[i])
                           for i in range(self.T)])
        self._C += PARAMS_LQG['cov_reg'] * np.identity(C.shape[-1])

    def update_policy(self, policy: Policy = None, file_path: str = None,
                      t_x=None, inv_t_u=None, cov: np.ndarray = None):
        if not isinstance(policy, Policy):
            policy = torch.load(file_path)
            policy.eval()

            def policy_mean(x):
                if callable(t_x):
                    x = t_x(x)
                x = torch.FloatTensor(x)
                out = policy(x.to(device)).detach().cpu().numpy()
                if callable(inv_t_u):
                    out = inv_t_u(out)
                return out
        else:
            policy.eval()

            def policy_mean(x):
                return policy.to_numpy(x, t_x=t_x, t_u=inv_t_u)

        self.policy_mean = policy_mean
        if isinstance(cov, np.ndarray):
            self.policy_cov = cov

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
                 nu=1e-3, lamb=None,
                 policy_mean=None,
                 F=1e-3
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
                Default: 1e-3.
            lamb : Langrange's multiplier, contraint weight.
                Default: None (np.array).
            F : Model noise p(x'|x, u).
                Default: 1e-3.

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
        self._F = F * np.identity(self.n_x)

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
        C = nearestPD(self.control._C[i])
        a1 = np.hstack([sigma, sigma @ K.T])
        a2 = np.hstack([K @ sigma, C + K @ sigma @ K.T])
        sigma = f_xu @ np.vstack([a1, a2]) @ f_xu.T + self._F
        return sigma

    def _dist_dynamics(self, xs, us):
        mu = xs[0]
        sigma = self._F
        N = us.shape[0]
        cov_dynamics = np.empty_like(self.cov_dynamics)
        for i in range(N):
            mu = self._mean_dynamics(xs[i], us[i], i=i, mu=mu)
            sigma = self._cov_dynamics(xs[i], us[i], i=i, sigma=sigma)
            self.mean_dynamics[i] = mu
            cov_dynamics[i] = sigma
        self._C = np.array([nearestPD(cov_dynamics[i]) for i in range(N)])

    def update_control(self, control):
        self.control = control

    def reg_cov(self, x, y):
        return y * np.exp(-y/20 * x)

    def _cost(self, x, u, i):
        cov_dynamics = self.cov_dynamics[i] + self.reg_cov(
            i, PARAMS_ONLINE['cov_reg']) * np.identity(x.shape[-1])
        # if not issymmetric(cov) or (np.linalg.eigvals(cov) < 0).any():
        #     breakpoint()
        c = 0.0
        c -= u.T@self.lamb[i]
        # log policy distribution
        c -= self.nu[i] * \
            multivariate_normal.logpdf(
                x=u, mean=self.policy_mean(x), cov=self.policy_cov,
                allow_singular=True)
        # log dynamics distribution
        c -= multivariate_normal.logpdf(x=x,
                                        mean=self.mean_dynamics[i],
                                        cov=cov_dynamics,
                                        allow_singular=True)
        return c


class PolicyNoise(object):

    def __init__(self, policy):
        self.policy = policy

    def get_action(self, action, t=0):
        return multivariate_normal.rvs(action, self.policy._sigma, 1)


def logpdf(x, mean, cov):
    '''
    https://blogs.sas.com/content/iml/2020/07/15/multivariate-normal-log-likelihood.html
    '''
    inv_cov = np.linalg.inv(cov)
    return (-1/2) * mahalanobis(x, mean, inv_cov) ** 2


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


def symmetrize(A: np.ndarray):
    '''Numerically symetrize a theoretical symetric matrix'''
    return (A + A.T) / 2.


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2]. Found on [3].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    [3] https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    Id = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += Id * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
