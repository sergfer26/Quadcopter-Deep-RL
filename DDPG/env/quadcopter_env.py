import numpy as np
import gym
from gym import spaces
from numpy import pi, tanh, sin, cos, tan
from numpy.linalg import norm
from scipy.integrate import odeint


# constantes
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

# ## Diseño del ambiente, step y reward
r2 = lambda z, w, epsilon : (1 - tanh(z/epsilon)) * w **2

# constantes del ambiente
Vel_Max = 60 #Velocidad maxima de los motores
Vel_Min = -20
Velang_Max = 10 
Velang_Min = -10
# du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz

Low_obs = np.array([0, 0, 0, Velang_Min, Velang_Min, Velang_Min, 0, 0, 0, -10, -10, -10])
High_obs = np.array([10, 10, 10, Velang_Max, Velang_Max, Velang_Max, 2 * pi, 2 * pi, 2 * pi, 18, 18, 18])


"""Quadcopter Environment that follows gym interface"""
class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Box(low = Vel_Min * np.ones(4), high = Vel_Max*np.ones(4))
        self.observation_space = spaces.Box(low = Low_obs, high = High_obs)
        self.i = 0
        self.rz, self.rw = [1,0.5] # perturbaciones
        self.state = self.reset()
        self.time_max = 17
        self.tam = 500 
        self.time = np.linspace(0, self.time_max, self.tam)
        self.z_e = 15
        self.epsilon = 0.1 # parametro reward
        self.beta = 3 # 
        self.flag = True

    def reward_f(self):
        z = self.state[-1]
        w = self.state[2]
        if z <= 0:
            return -1e2
        #else:
        #return -np.linalg.norm([z-self.z_e,2*w])
            #return -0.5*(abs(z-self.z_e) + abs(w))
        #return -10e-4*sqrt((z-self.z_e)**2 + w**2 )
        #return np.tanh(1 - 0.00005*(abs(self.state - np.array([15,0]))).sum()
        #if 13 < z < 17:
        #    import pdb; pdb.set_trace()
            
        #else:
        #    return 0
        else:
            return -np.linalg.norm([abs(z-self.z_e)**self.beta, r2(abs(z - self.z_e), 2*w, self.epsilon)])
            #return - abs(z-self.z_e) - r2(abs(z - self.z_e), w)
            #return -np.linalg.norm([z-self.z_e,w**2])
            #return -((z-self.z_e)**2 + r2(z-self.z_e, w))
            
        
    def is_done(self):
        z = self.state[-1]
        w = self.state[2]
        #Si se te acabo el tiempo
        if self.i == self.tam-2:
            return True
        #Si estas muy lejos
        #elif self.reward_f() < -1e3:
        #    return True
        #Si estas muy cerca
        #elif self.reward_f() > -1e-3:
        #    return True
        elif self.flag:
            if z < 0 or z > 18:
                return True
            else:
                return False
            #elif 0 < w < High_obs[1]:
           #  return True
          ##else:
            # return False

    def step(self,action):
        #import pdb; pdb.set_trace()
        w1, w2, w3, w4 = action
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(f, self.state, t, args=(w1, w2, w3, w4))[1]
        self.state = delta_y
        reward = self.reward_f()
        done = self.is_done()
        self.i += 1
        return delta_y, reward, done

    def reset(self):
        z, w = max(0, 15 + float(np.random.uniform(-self.rz,self.rz,1))) ,float(np.random.uniform(-self.rw,self.rw,1))
        self.state = np.array([0, 0, w, 0, 0, 0, 0, 0, 0, 0, 0, z])
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