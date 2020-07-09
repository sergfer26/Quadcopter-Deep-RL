import numpy as np
import gym
from gym import spaces
from numpy import pi, tanh, sin, cos, tan
from numpy.linalg import norm
from numpy.random import uniform as unif
from scipy.integrate import odeint


# constantes del ambiente
VEL_MAX = 60 #Velocidad maxima de los motores
VEL_MIN = -20
VELANG_MIN = -10
VELANG_MAX = 10
# du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz
LOW_OBS = np.array([-10, -10, -10, VELANG_MIN, VELANG_MIN, VELANG_MIN, -pi, -pi, -pi, 0, 0, 0])
HIGH_OBS = np.array([10, 10, 10, VELANG_MAX, VELANG_MAX, VELANG_MAX, pi, pi, pi, 22, 22, 22])
PSIE = 0.0; THETAE = 0.0; PHIE = 0.0
XE = 15.0; YE = 15.0; ZE = 15.00

# parametros de entrenamineto
BETA = 0.1
EPSILON = 0.1
#                   du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz
WEIHGTS = np.array([0.01, 0.01, 1, 1, 1, 0.005, 0.005, 1, 1, 0.01, 0.01, 1]) 
R = 1000

TIME_MAX = 17.00
STEPS = 500

G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))


# ## Sistema dinámico
def f(y, t, w1, w2, w3, w4):
    #El primer parametro es un vector
    #W,I tambien
    u, v, w, p, q, r, psi, theta, phi, x, y, z = y
    Ixx, Iyy, Izz = I
    W = np.array([w1, w2, w3, w4])
    du = r * v - q * w - G * sin(theta)
    dv = p * w - r * u - G * cos(theta) * sin(phi)
    dw = q * u - p * v + G * cos(phi) * cos(theta) - (K/M) * norm(W) ** 2
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B / Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * sin(phi) + r * cos(phi)) * (1 / cos(theta))
    dtheta = q * cos(phi) - r * sin(phi)
    dphi = p + (q * sin(phi) + r * cos(phi)) * tan(theta)
    dx = u
    dy = v
    dz = w
    return du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz


"""Quadcopter Environment that follows gym interface"""
class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box(low = VEL_MIN * np.ones(4), high = VEL_MAX * np.ones(4))
        self.observation_space = spaces.Box(low = LOW_OBS, high = HIGH_OBS)
        self.i = 0
        self.p = np.array([0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.goal = np.array([0, 0, 0, 0, 0, 0, PSIE, THETAE, PHIE, XE, YE, ZE])
        self.weights = WEIHGTS
        self.state = self.reset()
        self.old_state = self.state 
        self.time_max = TIME_MAX
        self.tam = STEPS
        self.time = np.linspace(0, self.time_max, self.tam)
        self.epsilon = EPSILON # parametro reward
        self.beta = BETA 
        self.flag = True
        self.umbral = 1

    def r(self):
        if norm(self.state[9:] - self.goal[9:]) < self.umbral:
            return R
        else:
            return R * (norm(self.old_state[9:] - self.goal[9:]) - norm(self.state[9:] - self.goal[9:])) / 10

    def get_reward(self):
        if LOW_OBS[-1] <  self.state[-1] < HIGH_OBS[-1]:
        #if self.observation_space.contains(self.state):
            state = np.copy(self.state); old_state = np.copy(self.old_state)
            state[6:9] = np.remainder(state[6:9], 2 * pi); old_state[6:9] = np.remainder(old_state[6:9], 2 * pi)
            state = np.multiply(state, self.weights); old_state = np.multiply(old_state, self.weights)
            d1 = norm(old_state[9:] - self.goal[9:]) - norm(state[9:] - self.goal[9:])
            d2 = norm(state[0:6])
            d3 = norm(state[6:9])
            return self.r() + tanh(1 - self.epsilon * (d1 ** 2).sum()) +  \
                tanh(1 - self.beta * (d2 ** 2).sum()) + tanh(1 - self.beta * (d3 ** 2).sum())
        else:
            return -1e4
        
    def is_done(self):
        #Si se te acabo el tiempo
        if self.i == self.tam-2:
            return True
        elif self.flag:
            # aux = [(x > l) & (x < h) for x, l, h in zip(self.state[9:], LOW_OBS[9:], HIGH_OBS[9:])]
            if LOW_OBS[-1] < self.state[-1] < HIGH_OBS[-1]: #all(aux):
                return False
            else:
                return True

    def step(self, action):
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f, self.state, t, args=(w1, w2, w3, w4))[1]
        self.state = delta_y
        reward = self.get_reward()
        done = self.is_done()
        self.i += 1
        return delta_y, reward, done

    def reset(self):
        self.i = 0
        self.state = np.array([max(0, g + unif(-e, e)) for e, g in zip(self.p, self.goal)])
        return self.state
        
    def render(self, mode='human', close=False):
        pass


def control_feedback(x, y, F):
    '''
    Realiza el control lineal de las velocidades W
    dadas las variables (x, y).
    param x: variable independiente (dx = y)
    param y: variable dependiente
    param F: matriz 2x4 de control
    regresa: W = w1, w2, w3, w4 
    '''
    A = np.array([x, y]).reshape((2, 1))
    return np.dot(F, A).reshape((4,))