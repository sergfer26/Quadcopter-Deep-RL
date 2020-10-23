from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
from numpy import cos,sin,tan,pi
from scipy.integrate import odeint
from numpy.linalg import norm

G = 9.81
I = (4.856 * 10 ** -3, 4.856 * 10 ** -3, 8.801 * 10 **-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))


TIME_MAX = 3.00
STEPS = 80

TIME = np.linspace(0, TIME_MAX, STEPS)

VELANG_MIN = -10
VELANG_MAX = 10

LOW_OBS = np.array([-10, -10, -10,  0, 0, 0, VELANG_MIN, VELANG_MIN, VELANG_MIN, -pi, -pi, -pi])
HIGH_OBS = np.array([10, 10, 10, 22, 22, 22, VELANG_MAX, VELANG_MAX, VELANG_MAX, pi, pi, pi])

W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0
W0_inf = W0*(0.5)
W0_sup = W0*(1.5)



# ## Sistema dinámico
def f(y, t, w1, w2, w3, w4):
    #El primer parametro es un vector
    #W,I tambien
    u, v, w, _, y, _, p, q, r, _, theta, phi = y
    Ixx, Iyy, Izz = I
    W = np.array([w1, w2, w3, w4])
    du = r * v - q * w - G * sin(theta)
    dv = p * w - r * u - G * cos(theta) * sin(phi)
    dw = q * u - p * v + G * cos(phi) * cos(theta) - (K/M) * norm(W) ** 2
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B/Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * sin(phi) + r * cos(phi)) * (1 / cos(theta))
    dtheta = q * cos(phi) - r * sin(phi)
    dphi = p + (q * sin(phi) + r * cos(phi)) * tan(theta)
    dx = u; dy = v; dz = w
    return du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi



def evaluate_model(A):
    w1, w2, w3, w4 = A
    state = np.zeros(12)
    state[3:6] = 15*np.ones(3)
    return odeint(f, state, TIME, args=(w1, w2, w3, w4))[-1]

    
problem = {'num_vars': 4, 'names': ['A1', 'A2', 'A3','A4'],
    'bounds':  [[x,y] for x,y in zip(W0_inf,W0_sup)]}

n = 10
param_values = saltelli.sample(problem, n)
Y = np.zeros((param_values.shape[0],12))

for i, A in enumerate(param_values):
    Y[i,:] = evaluate_model(A)

Si = sobol.analyze(problem, Y)


