import numpy as np
import gym
import numba
from Linear.equations import f, jac_f, angles2rotation, W0
from Linear.constants import CONSTANTS, omega_0, F, C
from numba import cuda
from gym import spaces
from numpy.linalg import norm
from scipy.integrate import odeint
from params import PARAMS_ENV, PARAMS_OBS


DT = PARAMS_ENV['dt']
STEPS = PARAMS_ENV['STEPS']

# constantes de la recompensa
K1 = eval(PARAMS_ENV['K1'])
K2 = eval(PARAMS_ENV['K2'])
K3 = eval(PARAMS_ENV['K3'])

# constantes del control
omega0_per = PARAMS_ENV['omega0_per']
VEL_MAX = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * omega0_per


# limites espaciales del ambiente
# u, v, w, x, y, z, p, q, r, psi, theta, phi
LOW_OBS = np.array([- eval(v) for v in PARAMS_OBS.values()])
HIGH_OBS = np.array([eval(v) for v in PARAMS_OBS.values()])

G = CONSTANTS['G']
M = CONSTANTS['M']
K = CONSTANTS['K']
B = CONSTANTS['B']
L = CONSTANTS['L']
Ixx = CONSTANTS['Ixx']
Iyy = CONSTANTS['Iyy']
Izz = CONSTANTS['Izz']

F1, F2, F3, F4 = F
c1, c2, c3, c4 = C


class QuadcopterEnv(gym.Env):
    '''Quadcopter Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, u0=None, low=LOW_OBS, high=HIGH_OBS):
        self.action_space = spaces.Box(
            low=VEL_MIN * np.ones(4), high=VEL_MAX * np.ones(4),
            dtype=np.float64)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float64)
        self.state = self.reset()  # estado interno del ambiente
        self.set_time(STEPS, DT)
        self.flag = False
        self.is_cuda_available()
        self.W0 = W0 if not isinstance(u0, np.ndarray) else u0

    def set_obs_space(self, low, high):
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float64
        )

    def is_cuda_available(self):
        '''
            Verifica si esta disponible el gpu,
            en caso de que sí, las funciones f y jac trabajaran sobre numba.
        '''
        if cuda.is_available():
            self.f = numba.jit(f)
            self.jac = numba.jit(jac_f)
        else:
            self.f = f
            self.jac = jac_f

    def is_contained(self, state):
        '''
            Verifica si los valores de (x, y, z) estan dentro
            de los rangos definidos por el ambiente.
            Params:
                * state (array): vector de 12 o 18 posiciones.
            Salidas:
                * (bool): estado de contención.
        '''
        x = state[3:6]
        high = self.observation_space.high[3:6]
        low = self.observation_space.low[3:6]
        aux = np.logical_and(low <= x, x <= high)
        return aux.all()

    def is_stable(self, state):
        '''
            Verifica el drone esta estable, cerca del objetivo.
            Params:
                * state (array): vector de 12 o 18 posiciones;
            Salidas:
                * (bool): estado de estabilidad.
        '''
        x = state[3:6]
        v = np.concatenate([state[0:3], state[6:9]])
        x_ = 1 * np.ones(3)
        v_ = 0.50 * np.ones(6)
        x_st = np.logical_and(- x_ <= x, x <= x_)
        v_st = np.logical_and(- v_ <= v, v <= v_)
        return x_st.all() and v_st.all()

    def get_score(self, state):
        '''
            Verifica el drone esta contenido y estable.
            Params:
                * state (array): vector de 12 o 18 posiciones.
            Salidas:
                * score1 (int): puntaje de contención.
                * score2 (int): puntahje de estabilidad.
        '''
        score1 = 0
        score2 = 0
        if self.is_contained(state[3:6]):
            score2 = 1
            if self.is_stable(state):
                score1 = 1
        return score1, score2

    def get_reward(self, state, action):
        '''
        falta
        '''
        # u, v, w, x, y, z, p, q, r, psi, theta, phi = state
        penalty = K1 * norm(state[3:6])
        mat = angles2rotation(state[9:], flatten=False)
        penalty += K2 * norm(np.identity(3) - mat)
        penalty += K3 * norm(action)
        return - penalty

    def is_done(self):
        '''
            is_done verifica si el drone ya termino de hacer su tarea;

            regresa valor booleano.
        '''
        if self.i == self.steps-1:  # Si se te acabo el tiempo
            return True
        elif self.flag:  # Si el drone esta estrictamente contenido
            if self.is_contained(self.state):
                return False
            else:
                return True
        else:
            return False

    def step(self, action):
        '''
        Realiza la interaccion entre el agente y el ambiente en
        un paso de tiempo.

        Argumentos
        ----------
        action: `np.ndarray`
            Representa la acción actual ($a_t$). Arreglo de dimesnión (4,) 
            con valores entre [low, high].

        Retornos
        --------
        state : `np.ndarray`
            Representa el estado siguiente ($s_{t+1}$). Arreglo de dimensión (12,).
        reward : `float`
            Representa la recompensa siguiente ($r_{t+1}$).
        done : `bool`
            Determina si la trayectoria ha terminado o no.
        info : `dict`
            Es la información  adicional que proporciona el sistema. 
            info['real_action'] : action.
        '''
        w1, w2, w3, w4 = action + self.W0
        t = [self.time[self.i], self.time[self.i+1]]
        y_dot = odeint(self.f, self.state, t, args=(w1, w2, w3, w4))[
            1]  # , Dfun=self.jac)[1]
        self.state = y_dot
        reward = self.get_reward(y_dot, action)
        done = self.is_done()
        self.i += 1
        info = {'real_action': action}
        return self.state, reward, done, info

    def reset(self):
        '''
            reset fija la condición inicial para cada simulación del drone

            regresa el estado actual del drone.
        '''
        self.i = 0
        self.state = self.observation_space.sample()
        return self.state

    def set_time(self, steps, dt):
        '''
        set_time fija la cantidad de pasos y el tiempo de simulación;
        steps: candidad de pasos (int);
        time_max: tiempo de simulación (float).
        '''
        self.dt = dt
        self.time_max = dt * steps
        self.steps = steps
        self.time = np.linspace(0, self.time_max, self.steps + 1)

    def render(self, mode='human', close=False):
        '''
            render hace una simulación visual del drone.
        '''
        pass
