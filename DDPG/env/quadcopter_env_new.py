import numpy as np
import gym
from gym import spaces
from numpy import pi, tanh, sin, cos, tan
from numpy.linalg import norm
from numpy.random import uniform as unif
from scipy.integrate import odeint


# constantes del ambiente
VEL_MAX = 55 #60 #Velocidad maxima de los motores
VEL_MIN = -15 #-20
VELANG_MIN = -10
VELANG_MAX = 10

# du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi
LOW_OBS = np.array([-10, -10, -10,  0, 0, 0, VELANG_MIN, VELANG_MIN, VELANG_MIN, -pi, -pi, -pi])
HIGH_OBS = np.array([10, 10, 10, 22, 22, 22, VELANG_MAX, VELANG_MAX, VELANG_MAX, pi, pi, pi])
PSIE = 0.0; THETAE = 0.0; PHIE = 0.0
XE = 0.0; YE = 0.0; ZE = 15.0

TIME_MAX = 30.00
STEPS = 800

G = 9.81
I = (4.856 * 10 ** -3, 4.856 * 10 ** -3, 8.801 * 10 **-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))


def D(angulos):
    z, y, x = angulos # psi, theta, phi
    R = np.array([
        [cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x), cos(z) * sin(y) * cos(x) + sin(z) * sin(x)],
        [sin(z) * cos(y), sin(z) * cos(y) * sin(x) + cos(z) * cos(x), sin(z) * sin(y) * cos(x) - cos(z) * sin(x)], 
        [- sin(y), cos(y) * sin(x), cos(y) * cos(x)]
    ])
    return R


def funcion(state):
    angulos = state[9:]
    state = state[0:9]
    orientacion = np.matrix.flatten(D(angulos))
    return np.concatenate([state, orientacion])


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
    # return du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz
    return du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi


"""Quadcopter Environment that follows gym interface"""
class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box(low = VEL_MIN * np.ones(4), high = VEL_MAX * np.ones(4))
        self.observation_space = spaces.Box(low = LOW_OBS, high = HIGH_OBS)
        self.i = 0
        self.p = np.array([0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.goal = np.array([0, 0, 0, XE, YE, ZE, 0, 0, 0, PSIE, THETAE, PHIE])
        self.state = self.reset()
        self.time_max = TIME_MAX
        self.tam = STEPS
        self.time = np.linspace(0, self.time_max, self.tam)
        self.flag = True
        self.lam = 1

    def get_reward(self,x):
        vels = x[0:3]
        state = x[3:6]
        #orientacion = x[9:].reshape((3,3))
        if LOW_OBS[-1] <  self.state[5] < HIGH_OBS[-1]:
            r = 0
            #if norm(orientacion - np.identity(3)) < 0.08:
            if norm(vels) < 5e-2:
                if norm(state - self.goal[3:6]) < 1:
                    r += 10
            #r += - 50 * norm(orientacion - np.identity(3)) - 6e-1 * norm(state - self.goal[3:6]) #1.2
            r  += - 8e-1 * norm(vels) - 2e-1 * norm(state - self.goal[3:6])
            return r 
        return -1e5
        
    def is_done(self):
        #Si se te acabo el tiempo
        if self.i == self.tam-2:
            return True
        elif self.flag:
            if LOW_OBS[-1] < self.state[5] < HIGH_OBS[-1]: #all(aux):
                return False
            else:
                return True
        else: 
            return False

    def step(self, action):
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f, self.state, t, args=(w1, w2, w3, w4))[1]
        self.state = delta_y
        transformacion = funcion(delta_y)
        reward = self.get_reward(transformacion)
        done = self.is_done()
        self.i += 1
        return transformacion, reward, done

    def reset(self):
        self.i = 0
        self.state = np.array([max(0, g + unif(-e, e)) for e, g in zip(self.p, self.goal)])
        return self.state
        
    def render(self, mode='human', close=False):
        pass

