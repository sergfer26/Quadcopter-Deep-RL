import numpy as np
import gym
from gym import spaces
from numpy import pi, tanh
from numpy.linalg import norm
from scipy.integrate import odeint


# constantes
G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))

VEL_MAX = 60 # Velocidad maxima de los motores
VEL_MIN = - 20
LOW_OBS = np.array([0, 0]) # z, w
HIGH_OBS = np.array([22, 22])
ZE = 15.00
BETA = 0.1
EPSILON = 0.1
R = 1000

TIME_MAX = 17.00
STEPS = 500
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0

# ## Sistema dinámico
def f(y, t, w1, w2, w3, w4):
    # El primer parametro es un vector
    # W tambien
    z, w = y
    dz = w
    W = [w1, w2 , w3 , w4]
    dw = G - (K/M) * norm(W) ** 2
    # dw = (-2*K/M)*omega_0*(w1 + w2 + w3 + w4)
    return dz, dw

# ## Diseño del ambiente, step y reward
r2 = lambda z, w, epsilon : (1 - tanh(z/epsilon)) * w **2


"""Quadcopter Environment that follows gym interface"""
class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box(low = VEL_MIN * np.ones(1), high = VEL_MAX * np.ones(1))
        self.observation_space = spaces.Box(low = LOW_OBS, high = HIGH_OBS)
        self.i = 0
        self.time_max = TIME_MAX
        self.tam = STEPS
        self.time = np.linspace(0, self.time_max, self.tam)
        self.rz, self.rw = [1, 0.5]
        self.z0 = max(0, 15 + float(np.random.uniform(-self.rz, self.rz, 1)))
        self.w = float(np.random.uniform(-self.rw, self.rw, 1))
        self.state = self.reset() #z_t, z_t+1        
        self.z_e = 15
        self.epsilon = EPSILON
        self.beta = BETA
        self.flag = True

    def reward_f(self):
        #import pdb;pdb.set_trace()
        oz,z = self.state
        if LOW_OBS[1] < z < HIGH_OBS[1]:
            u = abs(oz - self.z_e) - abs(z - self.z_e)
            v = abs(z - self.z_e)
            # = (z-oz)/(17/500)
            #return -np.linalg.norm([10*abs(z-self.z_e)**self.epsilon, r2(abs(z - self.z_e), 2*w,self.epsilon)])
            #return - abs(z-self.z_e) - r2(abs(z - self.z_e), w)
            #return -np.linalg.norm([z-self.z_e,w**2])
            #return -((z-self.z_e)**2 + r2(z-self.z_e, w))
            return tanh(1 - self.epsilon * (u ** 2).sum()) + tanh(1 - self.beta * (v ** 2).sum())
            # return -abs(w) - abs(z-15)
        else:
            return -1e4
            
    def is_done(self):
        _, z = self.state
        #Si se te acabo el tiempo
        if self.i == self.tam-3:
            return True
        elif self.flag:
            if z < LOW_OBS[1] or z > HIGH_OBS[1]:
                return True
            else:
                return False

    def step(self,action):
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f,[self.state[0], self.w], t, args=(w1, w2, w3, w4))[1]
        self.state = np.array([self.state[-1], delta_y[0]])
        self.w = delta_y[1]
        reward = self.reward_f()
        done = self.is_done()
        self.i += 1
        return self.state, reward, done

    def reset(self):
        w1, w2, w3, w4 = W0
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f,[self.z0,self.w], t, args=(w1, w2, w3, w4))[1]
        self.i = 0
        self.state = np.array([self.z0,delta_y[0]])
        self.w = delta_y[1]
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
