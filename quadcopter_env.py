import numpy as np
import gym
import numba
import torch
from numba import cuda
from gym import spaces
from numpy import pi, sin, cos, tan
from numpy.linalg import norm
from scipy.integrate import odeint


TIME_MAX = 30.00
STEPS = 800

# perturbar constantes 10 %
G = 9.81
I = (4.856 * 10 ** -3, 4.856 * 10 ** -3, 8.801 * 10 **-3) # perturbar momentos
B, M, L = 1.140*10**(-6), 1.433, 0.225 
K = 0.001219  # kt
omega_0 = np.sqrt((G * M)/(4 * K))
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0

# constantes del ambiente
VEL_MAX = omega_0 * 0.40  #60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * 0.40 
VELANG_MIN = -0.2
VELANG_MAX = 0.2

# du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi
LOW_OBS = np.array([-0.2, -0.2, -0.2,  -5, -5, -5, VELANG_MIN, VELANG_MIN, VELANG_MIN, -pi/8, -pi/8, -pi/8])
HIGH_OBS = np.array([0.2, 0.2, 0.2, 5, 5, 5, VELANG_MAX, VELANG_MAX, VELANG_MAX, pi/8, pi/8, pi/8])
PSIE = 0.0; THETAE = 0.0; PHIE = 0.0
XE = 0.0; YE = 0.0; ZE = 0.0


sec = lambda x: 1/cos(x)


def rotation_matrix(angles):
    '''
        Obtine la matriz de rotación
    '''
    z, y, x = angles # psi, theta, phi
    R = np.array([
        [cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x), 
        cos(z) * sin(y) * cos(x) + sin(z) * sin(x)],
        [sin(z) * cos(y), sin(z) * cos(y) * sin(x) + cos(z) * cos(x), 
        sin(z) * sin(y) * cos(x) - cos(z) * sin(x)], 
        [- sin(y), cos(y) * sin(x), cos(y) * cos(x)]
    ])
    return R


def jac_f(X, t, w1, w2, w3, w4):
    Ixx, Iyy, Izz = I
    a1 = (Izz - Iyy)/Ixx
    a2 = (Ixx - Izz)/Iyy
    u, v, w, _, _, _, p, q, r, _, theta, phi = X
    J = np.zeros((12, 12))
    ddu = np.zeros(12); J[0, 1:3] = [r, -q]
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
        self.action_space = spaces.Box(low=VEL_MIN * np.ones(4), high=VEL_MAX * np.ones(4))
        self.observation_space = spaces.Box(low=LOW_OBS, high=HIGH_OBS)
        self.p = np.zeros(12)
        self.goal = np.array([0, 0, 0, XE, YE, ZE, 0, 0, 0, PSIE, THETAE, PHIE])
        self.state = self.reset()
        self.set_time(STEPS, TIME_MAX)
        self.flag = False
        self.is_cuda_available()

    def is_cuda_available(self):
        if cuda.is_available():
            self.f = numba.jit(f)
            self.jac = numba.jit(jac_f)
        else:
            self.f = f
            self.jac = jac_f

    def is_contained(self, state):
        x = state[3:6]
        # x = state[3:6]
        high = self.observation_space.high[3:6]
        low = self.observation_space.low[3:6]
        aux = np.logical_and(low <= x, x <= high)
        return aux.all()

    def is_stable(self, state):
        x = state[3:6]
        v = np.concatenate([state[0:3], state[6:9]])
        x_ = 1 * np.ones(3)
        v_ = 0.50 * np.ones(6)
        x_st = np.logical_and(self.goal[3:6] - x_ <= x, x <= self.goal[3:6] + x_)
        v_st = np.logical_and(-v_ <= v, v <= v_)
        return x_st.all() and v_st.all()

    def get_score(self, state):
        score1 = 0
        score2 = 0
        if self.is_contained(state[3:6]):
            score2 = 1
            vel = np.concatenate([state[0:3], state[6:9]])
            if self.is_stable(state[3:6], vel):
                score1 = 1
        return score1, score2

    def get_reward(self, state):
        x = state[3:6]
        x_ = 5 * np.ones(3)
        r = 0.0
        vel = np.concatenate([state[0:3], state[6:9]])
        x_st = np.logical_and(self.goal[3:6] - x_ <= x, x <= self.goal[3:6] + x_)
        if x_st.all():
            r = 1
        d1 = norm(x - self.goal[3:6])
        d2 = norm(vel)
        d3 = norm(rotation_matrix(state[9:]))
        return r - (0.01 * d2 + 0.01 * d1)

    def is_done(self):
        #Si se te acabo el tiempo
        if self.i == self.steps-2:
            return True
        elif self.flag:
            if self.is_contained(self.state[3:6]): 
                return False
            else:
                return True
        else: 
            return False

    def step(self, action):
        breakpoint()
        w1, w2, w3, w4 = action + W0
        t = [self.time[self.i], self.time[self.i+1]]
        delta_y = odeint(self.f, self.state, t, args=(w1, w2, w3, w4))[1]#, Dfun=self.jac)[1]
        self.state = delta_y # estado interno del ambiente 
        reward = self.get_reward(delta_y)
        done = self.is_done()
        self.i += 1
        return self.state, reward, done

    def reset(self):
        self.i = 0
        return self.observation_space.sample()

    def set_time(self, steps, time_max):
        self.time_max = time_max
        self.steps = steps
        self.time = np.linspace(0, self.time_max, self.steps)
        
    def render(self, mode='human', close=False):
        pass


class AgentEnv(gym.ActionWrapper, gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate((self.observation_space.low[:9], - np.ones(9)))
        high = np.concatenate((self.observation_space.high[:9], np.ones(9)))
        self.observation_space = spaces.Box(low=low, high=high)

    def observation(self, obs):
        angles = obs[9:]
        ori = np.matrix.flatten(rotation_matrix(angles))
        return np.concatenate([obs[0:9], ori])

    def reverse_observation(self, obs):
        mat = obs[9:].reshape((3, 3))
        psi = np.arctan(mat[1, 0]/mat[0, 0])
        theta = np.arctan(- mat[2, 0]/np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        phi = np.arctan(mat[2, 1] / mat[2, 2])
        angles = np.array([psi, theta, phi])
        return np.concatenate([obs[0:9], angles])

    def action(self, action):
        '''
        se alimenta de la tanh
        '''
        high = self.action_space.high
        low = self.action_space.low
        if torch.is_tensor(action):
            high = torch.tensor(high)
            low = torch.tensor(low)

        act_k = (high - low)/ 2.
        act_b = (high + low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        high = self.action_space.high
        low = self.action_space.low
        if torch.is_tensor(action):
            high = torch.tensor(high)
            low = torch.tensor(low)

        act_k_inv = 2./(high - low)
        act_b = (high + low)/ 2.
        return act_k_inv * (action - act_b)

    def step(self, action):
        state, reward, done = super().step(action)
        state = self.observation(state)
        return state, reward, done

    def reset(self):
        state = super().reset()
        return self.observation(state)



