'''
 Quadrotor dynamics
'''
from typing import Any
import numpy as np
import torch
from numpy.linalg import norm
from params import PARAMS_ENV
# from ilqr import iLQR
# from ilqr.utils import GetSyms, Constrain
# from ilqr.containers import Dynamics, Cost
from scipy.spatial.transform import Rotation
from Linear.equations import rotation2angles, angles2rotation
from utils import wrap_angle


G = 9.81
Ixx, Iyy, Izz = 4.856 * 1e-3, 4.856 * 1e-3, 8.801 * \
    1e-3  # (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140 * 1e-6, 1.433, 0.225
K = 0.001219  # 2.980 * 1e-6  # kt
omega_0 = np.sqrt((G * M)/(4 * K))
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0

K1 = eval(PARAMS_ENV['K1'])
K11 = eval(PARAMS_ENV['K11'])
K2 = eval(PARAMS_ENV['K2'])
K21 = eval(PARAMS_ENV['K21'])
K3 = eval(PARAMS_ENV['K3'])

omega0_per = .60
VEL_MAX = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * omega0_per


class ReferenceReward(object):

    def __init__(self, mask=None, action_weight=1e-1, **kwargs) -> None:
        '''
        Parameters
        ----------
        kwarg : `dict`
            Each keyword argument represents the weight value of corresponding 
            variable.
            - lamb_x : `float`
                weight for variable x 
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.weights = np.array(list(vars(self).values()))
        self.action_weight = action_weight
        self._states = None
        self._actions = None
        self._mask = mask

    def set_reference(self, states: np.ndarray, actions: np.ndarray):
        indices = np.array([-3, -2, -1])
        states[indices] = wrap_angle(states[indices])
        self._states = states
        self._actions = actions

    def __call__(self, state: np.ndarray, action: np.ndarray, i: int):
        indices = np.array([-3, -2, -1])
        state[indices] = wrap_angle(state[indices])
        if isinstance(self._mask, np.ndarray) or isinstance(self._mask, list):
            out = (self.weights *
                   (self._states[i][self._mask] - state[self._mask])).sum()
        else:
            out = (self.weights * (self._states[i] - state)).sum()

        out_action = self.action_weight * (self._actions[i] - action).sum()
        return out + out_action


# Quadcopter dynamics
def f(state, actions):  # Sistema dinámico
    '''
        f calcula el vector dot_x = f(x, t, w) (sistema dinamico);

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regresa dot_x
    '''
    u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    # Ixx, Iyy, Izz = I
    w1, w2, w3, w4 = actions + W0
    du = r * v - q * w - G * np.sin(theta)
    dv = p * w - r * u - G * np.cos(theta) * np.sin(phi)
    dw = q * u - p * v + G * \
        np.cos(phi) * np.cos(theta) - (K/M) * \
        (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B/Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * np.sin(phi) + r * np.cos(phi)) * (1.0 / np.cos(theta))
    dtheta = q * np.cos(phi) - r * np.sin(phi)
    dphi = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    dx = u
    dy = v
    dz = w
    return np.array([du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi])


# @njit
def transform_x(x):
    '''
        x_t -> o_t
    '''
    r = angles2rotation(x[9:])
    z = np.zeros(18)
    z[0:9] = x[0:9]
    z[9:18] = r
    return z


def inv_transform_x(x):
    '''
        o_t -> x_t
    '''
    r = x[9:]
    angles = rotation2angles(r)
    x[9:12] = angles
    return x[:12]


# @njit
def transform_u(u: np.ndarray, high=VEL_MAX, low=VEL_MIN):
    '''
        u_t -> a_t
    '''
    u = np.clip(u, a_min=VEL_MIN, a_max=VEL_MAX)
    act_k_inv = 2./(high - low)
    act_b = (high + low) / 2.
    return act_k_inv * (u - act_b)


def inv_transform_u(u: np.ndarray, high=VEL_MAX, low=VEL_MIN):
    '''
        a_t -> u_t

        action transforma una accion de valores entre [-1, 1] a
        los valores [low, high];
        action: arreglo de 4 posiciones con valores entre [-1, 1];
        regresa una acción entre [low, high].
    '''
    if torch.is_tensor(u):
        high = torch.tensor(high)
        low = torch.tensor(low)
    act_k = (high - low) / 2.
    act_b = (high + low) / 2.
    action = act_k * u + act_b
    return action


def signed_distance(X, A):
    return np.dot(X, A) / norm(A)


def my_cost(state, action, i):
    return K3 * norm(action) + terminal_cost(state, i)


def terminal_cost(state, i):
    # d_safe = 1.0
    u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    penalty = K1 * norm(z)
    penalty += 0.01 * norm(np.array([.1 - u, .1 - v, w]))
    penalty += 0.01 * norm(np.array([p, q, r]))

    mat = angles2rotation(np.array([psi, theta, phi]), flatten=False)
    penalty += K2 * norm(np.identity(3) - mat)

    # d1 = signed_distance([x, y, z, 1], [2, 1, 0, -2.5])
    # penalty += 0.1 * max(d_safe - d1, 0)
    # d2 = signed_distance([x, y, z, 1], [2, 1, 0, 2.5])
    # penalty += 0.1 * max(d_safe - d2, 0)
    penalty += 0.1 * abs(signed_distance([x, y, 1], [2, 1, 3]))
    return penalty


def penalty(state, action, i=None):
    '''
    falta
    '''
    # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    return terminal_penalty(state, i) + K3 * norm(action)


def terminal_penalty(state, i=None):
    # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    penalty = K1 * norm(state[3:6])
    penalty += K11 * norm(state[:3])
    mat = angles2rotation(state[9:], flatten=False)
    penalty += K21 * norm(state[6:9])
    penalty += K2 * norm(np.identity(3) - mat)
    return penalty


if __name__ == "__main__":
    pass
