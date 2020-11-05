import numpy as np
import gym
from gym import spaces
from numpy import pi, tanh, sin, cos, tan
from numpy.linalg import norm
from numpy.random import uniform as unif
from scipy.integrate import odeint


TIME_MAX = 30.00
STEPS = 800

# perturbar constantes 10 %
G = 9.81
I = (4.856 * 10 ** -3, 4.856 * 10 ** -3, 8.801 * 10 **-3) # perturbar momentos
B, M, L = 1.140*10**(-6), 1.433, 0.225 
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))

# constantes del ambiente
VEL_MAX = omega_0 * 0.20  #60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * 0.20
VELANG_MIN = -10
VELANG_MAX = 10

# du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi
LOW_OBS = np.array([-10, -10, -10,  0, 0, 0, VELANG_MIN, VELANG_MIN, VELANG_MIN, -pi, -pi, -pi])
HIGH_OBS = np.array([10, 10, 10, 30, 30, 30, VELANG_MAX, VELANG_MAX, VELANG_MAX, pi, pi, pi])
PSIE = 0.0; THETAE = 0.0; PHIE = 0.0
XE = 15.0; YE = 15.0; ZE = 15.0


sec = lambda x: 1/cos(x)


def D(angulos):
    '''
        Obtine la matriz de rotación
    '''
    z, y, x = angulos # psi, theta, phi
    R = np.array([
        [cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x), 
        cos(z) * sin(y) * cos(x) + sin(z) * sin(x)],
        [sin(z) * cos(y), sin(z) * cos(y) * sin(x) + cos(z) * cos(x), 
        sin(z) * sin(y) * cos(x) - cos(z) * sin(x)], 
        [- sin(y), cos(y) * sin(x), cos(y) * cos(x)]
    ])
    return R


def funcion(state):
    '''
        Obtiene un estado de 18 posiciones
    '''
    angulos = state[9:]
    state = state[0:9]
    orientacion = np.matrix.flatten(D(angulos))
    return np.concatenate([state, orientacion])


def jac_f(X, t, w1, w2, w3, w4):
    Ixx, Iyy, Izz = I
    a1 = (Izz - Iyy)/Ixx
    a2 = (Ixx - Izz)/Iyy
    u, v, w, _, _, _, p, q, r, _, theta, phi = X

    ddu = np.zeros(12); ddu[1:3] = [r, -q]
    ddv = [0, r, -q, 0, 0, 0, w, 0, -u, G * sin(theta) * sin(phi), -G * cos(theta) * cos(phi)]
    ddw = [q, -p, 0, 0, 0, 0, -v, u, 0, 0, G * sin(theta) * cos(phi), -G * cos(theta) * sin(phi)]
    ddx = np.zeros(12); ddx[0] = 1
    ddy = np.zeros(12); ddx[1] = 1
    ddz = np.zeros(12); ddx[2] = 1
    ddp = np.zeros(12); ddp[7] = -r * a1; ddp[8] = -q * a1
    ddq = np.zeros(12); ddq[6] = -r * a2; ddq[8] = -p * a2
    ddr = np.zeros(12)

    ddpsi = np.zeros(12); ddpsi[7:9] = [sin(phi), cos(phi) * sec(theta)]
    ddpsi[10:] = [r * cos(phi) * tan(theta) * sec(theta), (q * cos(phi) - r * sin(phi)) * sec(theta)]
    
    ddtheta = np.zeros(12); ddtheta[7:9] = [cos(phi), -sin(phi)]
    ddtheta[-1] =  -q * sin(phi) -r * cos(phi)

    ddphi = np.zeros(12); ddphi[6:9] = [1, sin(phi) * tan(theta), cos(phi) * tan(theta)]
    ddphi[10:] = [(q * sin(phi) + r * cos(phi)) * sec(theta) ** 2, (q * cos(phi) - r * sin(phi)) * tan(theta)]

    J = np.array([ddu, ddv, ddw, ddx, ddy, ddz, ddp, ddq, ddr, ddpsi, ddtheta, ddphi])
    return J


# ## Sistema dinámico
def f(X, t, w1, w2, w3, w4):
    #El primer parametro es un vector
    #W,I tambien
    u, v, w, _, _, _, p, q, r, _, theta, phi = X
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
        self.p = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.goal = np.array([0, 0, 0, XE, YE, ZE, 0, 0, 0, PSIE, THETAE, PHIE])
        self.state = self.reset()
        self.set_time(STEPS, TIME_MAX)
        self.flag = True
        self.d1 = 0.50

    def is_contained(self, x):
        # x = state[3:6]
        high = self.observation_space.high[3:6]
        low = self.observation_space.low[3:6]
        aux = np.logical_and(low <= x, x <= high)
        return aux.all()

    def is_stable(self, x, v):
        x_ = 0.20 * np.ones(3)
        v_ = 0.50 * np.ones(3)
        x_st = np.logical_and(self.goal[3:6] - x_ <= x, x <= self.goal[3:6] + x_)
        v_st = np.logical_and(self.goal[0:3] - v_ <= v, v <= self.goal[0:3] + v_)
        return x_st.all() and v_st.all()

    def get_score(self, state):
        score1 = 0
        score2 = 0
        if self.is_contained(state[3:6]):
            score2 = 1
            if self.is_stable(state[3:6], state[0:3]):
                score1 = 1
        return score1, score2

    def get_reward(self, state):
        x = state[3:6]
        orientacion = state[9:].reshape((3,3))
        if self.is_contained(x):
            d2 =  norm(orientacion - np.identity(3))
            d1 = norm(x - self.goal[3:6]) 
            if d1 < self.d1:
                return 1 -(10 * d2)
            else:
                return -(10 * d2 + d1)
        elif self.flag:
            return - 1e5
        else:
            return - 100
        
    def is_done(self):
        #Si se te acabo el tiempo
        if self.i == self.tam-2:
            return True
        elif self.flag:
            if self.is_contained(self.state[3:6]): 
                return False
            else:
                return True
        else: 
            return False

    def step(self, action):
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f, self.state, t, args=(w1, w2, w3, w4))[1]
        self.state = delta_y # estado interno del ambiente
        transformacion = funcion(delta_y) # estado del agente
        reward = self.get_reward(transformacion)
        done = self.is_done()
        self.i += 1
        return transformacion, reward, done

    def reset(self):
        self.i = 0
        self.state = np.array([g + unif(-e, e) for e, g in zip(self.p, self.goal)])
        # self.state[5] = max(0, self.state[5])
        return self.state

    def set_time(self, steps, time_max):
        self.time_max = time_max
        self.tam = steps
        self.time = np.linspace(0, self.time_max, self.tam)
        
    def render(self, mode='human', close=False):
        pass

