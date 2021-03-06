import numpy as np
import gym
from gym import spaces
from numpy import pi, tanh, exp
from numpy.linalg import norm
from scipy.integrate import odeint


# constantes del ambiente
VEL_MAX = 60 #Velocidad maxima de los motores
VEL_MIN = -20
LOW_OBS = 0 # np.array([0,-10])#z,w
HIGH_OBS = 22 # np.array([20, 10])
ZE = 15.00
BETA = 0.9
EPSILON = 0.1
R = 10

TIME_MAX = 17.00
STEPS = 500

G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))

dt = TIME_MAX/STEPS

# ## Sistema dinámico
def f(y, t, w1, w2, w3, w4):
    #El primer parametro es un vector
    #W tambien
    _, w = y
    dz = w
    W = [w1, w2 , w3 , w4]
    dw = G - (K/M) * norm(W) ** 2
    #dw = (-2*K/M)*omega_0*(w1 + w2 + w3 + w4)
    return dz, dw

# ## Diseño del ambiente, step y reward
r2 = lambda z, w, epsilon : (1 - tanh(z/epsilon)) * w ** 2


"""Quadcopter Environment that follows gym interface"""
class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box(low=VEL_MIN * np.ones(1), high=VEL_MAX * np.ones(1))
        self.observation_space = spaces.Box(low=LOW_OBS * np.ones(1), high=HIGH_OBS * np.ones(1))
        self.i = 0
        self.rz, self.rw = 1, 0.5 # perturbaciones
        self.state = self.reset()
        self.old_state = self.state 
        self.time_max = TIME_MAX
        self.tam = STEPS
        self.time = np.linspace(0, self.time_max, self.tam)
        self.z_e = ZE
        self.epsilon = EPSILON # parametro reward
        self.beta = BETA # 
        self.flag = True
        self.umbral = 1.0

    def r1(self, oz, z):
        if abs(z - ZE) < self.umbral:
            return R
        else:
            return 0

    def reward_f(self):
        z, _ = self.state
        oz, _ = self.old_state
        if LOW_OBS < z < HIGH_OBS:
            u = abs(oz - ZE) - abs(z - ZE)
            v = abs(z - ZE)
            # return self.r1(oz, z) - norm([abs(z - self.z_e) ** self.beta, r2(abs(z - self.z_e), 2 * abs(oz - z), self.epsilon)])
            return self.r1(oz, z) + tanh(1 - self.epsilon * (u ** 2).sum()) + tanh(1 - self.beta * (v ** 2).sum())
        else:
            return - 1e3

    def is_done(self):
        z, _ = self.state
        if self.i == self.tam-2:
        # Si se te acabo el tiempo
            return True
        elif self.flag:
            if LOW_OBS < z < HIGH_OBS:
                return False
            else:
                return True
        else:
            return False

    def step(self,action):
        #import pdb; pdb.set_trace()
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f, self.state, t, args=(w1, w2, w3, w4))[1]
        self.old_state = np.copy(self.state)
        self.state = delta_y
        reward = self.reward_f()
        done = self.is_done()
        self.i += 1
        return self.state, reward, done

    def reset(self):
        self.state = np.array([max(0, ZE + float(np.random.uniform(-self.rz,self.rz,1))) ,float(np.random.uniform(-self.rw,self.rw,1))])
        #self.state = np.array([10 ,0])
        #self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
        self.i = 0
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