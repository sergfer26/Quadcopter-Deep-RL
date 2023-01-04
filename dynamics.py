'''
 Quadrotor dynamics
'''
import numpy as np
from numpy.linalg import norm
from params import PARAMS_ENV
# from ilqr import iLQR
# from ilqr.utils import GetSyms, Constrain
# from ilqr.containers import Dynamics, Cost
from scipy.spatial.transform import Rotation
from Linear.equations import rotation2angles, angles2rotation


G = 9.81
Ixx, Iyy, Izz = 4.856 * 1e-3, 4.856 * 1e-3, 8.801 * \
    1e-3  # (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140 * 1e-6, 1.433, 0.225
K = 0.001219  # 2.980 * 1e-6  # kt
omega_0 = np.sqrt((G * M)/(4 * K))
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0

K1 = PARAMS_ENV['K1']
K2 = PARAMS_ENV['K2']
K3 = PARAMS_ENV['K3']

omega0_per = .60
VEL_MAX = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * omega0_per


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
def transform_x(X):
    r = angles2rotation(X[9:])
    z = np.zeros(18)
    z[0:9] = X[0:9]
    z[9:18] = r
    return z


def inv_transform_x(X):
    r = X[9:]
    angles = rotation2angles(r)
    X[9:12] = angles
    return X[:12]


# @njit
def transform_u(u: np.ndarray, high=VEL_MAX, low=VEL_MIN):
    u = np.clip(u, a_min=VEL_MIN, a_max=VEL_MAX)
    act_k_inv = 2./(high - low)
    act_b = (high + low) / 2.
    return act_k_inv * (u - act_b)


def inv_transform_u(u: np.ndarray, high=VEL_MAX, low=VEL_MIN):
    '''
        action transforma una accion de valores entre [-1, 1] a
        los valores [low, high];
        action: arreglo de 4 posiciones con valores entre [-1, 1];
        regresa una acción entre [low, high].
    '''
    act_k = (high - low) / 2.
    act_b = (high + low) / 2.
    action = act_k * u + act_b
    return action


def penalty(state, action, i):
    '''
    falta
    '''
    # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    penalty = K1 * norm(state[3:6])
    mat = angles2rotation(state[9:], flatten=False)
    penalty += K2 * norm(np.identity(3) - mat)
    penalty += K3 * norm(action)
    return penalty


def teminal_penalty(state, i):
    # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
    penalty = K1 * norm(state[3:6])
    mat = angles2rotation(state[9:], flatten=False)
    penalty += K2 * norm(np.identity(3) - mat)
    return penalty


if __name__ == "__main__":
    pass
