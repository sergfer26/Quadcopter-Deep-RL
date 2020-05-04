import numpy as np
import gym
from gym import spaces
from numpy import pi
#!/usr/bin/env python3
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from numpy.linalg import norm

G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219  # kt


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


Velocidad_Max = 50
#u, v, w, p, q, r, psi, theta, phi, x, y, z
Low_obs = np.array([-5,-5,-5,-pi/100,-pi/100,-pi/100,-pi/100,-pi/100,-pi/100,-5,-5,0])
High_obs = np.array(5,5,5,pi/100,pi/100,pi/100,pi/100,pi/100,pi/100,5,5,40])

Tiempo_Max = 30
tam = 1500

def reward_f(state,z_e):
    w, p, q, r, psi, theta, phi,z = state
    if z <= 0:
        return -1e4
    else:
        return  -np.linalg.norm([w, p, q, r, psi, theta, phi,z-z_e])

def is_done(state,z_e,t):
    #Si se te acabo el tiempo
    if t == 30:
        return True
    #Si tu reward es muy negativo
    elif reward_f(state,z_e) < -1e3:
        return True
    elif reward_f(state,z_e) > -1e-3:
        return True
    else:
        return False
        
class QuadcopterEnv(gym.Env):
  """Quadcopter Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(QuadCopterEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Box(low = np.zeros(4), high = Velocidad_Max*np.ones(4))
    # Example for using image as input:
    self.observation_space = spaces.Box(low = Low_obs, high = High_obs)
    self.t = 0
    
    def reward_f(self,state,z_e):
        w, p, q, r, psi, theta, phi,z = state
        if z <= 0:
            return -1e4
        else:
            return  -np.linalg.norm([w, p, q, r, psi, theta, phi,z-z_e])

    def is_done(self,state,z_e):
        #Si se te acabo el tiempo
        if self.t == 30:
            return True
        elif reward_f(state,z_e) < -1e3:
            return True
        elif reward_f(state,z_e) > -1e-3:
            return True
        else:
            return False

  def step(self, action,y_t,t,z_e):
      '''
    Obtiene una solución numérica de dy=f(y) para un tiempo t+1

    param W: arreglo de velocidades de los  4 rotores
    param y: arreglo de 12 posiciones del cuadricoptero
    param t: un intervalo de tiempo

    regresa: y para el siguiente paso de tiempo,el reward
    '''
    # Execute one time step within the environment
    w1, w2, w3, w4 = action
    delta_y = odeint(f, y_t, t, args=(w1, w2, w3, w4))[1]
    _, _, w, p, q, r, psi, theta, phi, _, _, z = delta_y
    reward = self.reward_f([w, p, q, r, psi, theta, phi,z],z_e)
    done = self.is_done([w, p, q, r, psi, theta, phi,z],z_e)
    self.t = t[1]
    return delta_y,reward,done
      
      
  def reset(self):
    # Reset the state of the environment to an initial state
      self.t = 0 
  def render(self, mode='human', close=False):
      pass 
      
    # Render the environment to the screen
