import numpy as np
import gym
import numba
from sympy.parsing.latex import parse_latex
from Linear.equations import rotation_matrix, f, jac_f, omega_0, W0
from DDPG.utils import OUNoise
from numba import cuda
from gym import spaces
from numpy.linalg import norm
from scipy.integrate import odeint
from params import PARAMS_ENV, PARAMS_OBS


TIME_MAX = PARAMS_ENV['TIME_MAX']
STEPS = PARAMS_ENV['STEPS']
FLAG = PARAMS_ENV['FLAG']
REWARD = PARAMS_ENV['reward']


# constantes del ambiente
omega0_per = PARAMS_ENV['omega0_per']
VEL_MAX = omega_0 * omega0_per  # 60 #Velocidad maxima de los motores 150
VEL_MIN = - omega_0 * omega0_per


# limites espaciales del ambiente
# u, v, w, x, y, z, p, q, r, psi, theta, phi
LOW_OBS = np.array([- v for v in PARAMS_OBS.values()])
HIGH_OBS = np.array([v for v in PARAMS_OBS.values()])


'''Quadcopter Environment that follows gym interface'''


class Reward(object):

    def __init__(self, tag):
        self.tag = tag
        if self.tag == 'r1':
            self.str = r'1 - 0.2 \sqrt{ x^2 + y^2 + z^2 }'
        elif self.tag == 'r2':
            self.str = r'4 \times 10-3 | X | + 2 \times 10 ^ {-4} | A | + 3 \times 10 ^ {-4} |\omega | + 5 \times 10 ^ {-4} | V|'
        elif self.tag == 'r3':
            self.str = r'\max(0, 1 - |X|) - 0.02 |\theta| - 0.03 |\omega|'

        self.expr = parse_latex(self.str)

    def eval(self, state, action):
        u, v, w, x, y, z, p, q, r, psi, theta, phi = state
        a1, a2, a3, a4 = action
        r = 0.0
        X = norm([x, y, z])
        omega = norm([p, q, r])
        theta = norm([psi, theta, phi])
        if self.tag == 'r1':
            r = self.expr.evalf(3, subs=dict(x=x, y=y, z=z))
        elif self.tag == 'r2':
            A = norm([a1, a2, a3, a4])
            V = norm([u, v, w])
            r = self.expr.evalf(3, subs=dict(X=X, A=A, V=V, omega=omega))
        elif self.tag == 'r3':
            r = self.expr.evalf(3, subs=dict(X=X, theta=theta, omega=omega))

        return float(r)


class QuadcopterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, reward=REWARD):
        self.action_space = spaces.Box(
            low=VEL_MIN * np.ones(4), high=VEL_MAX * np.ones(4),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            low=LOW_OBS, high=HIGH_OBS, dtype=np.float32)
        self.state = self.reset()  # estado interno del ambiente
        self.set_time(STEPS, TIME_MAX)
        self.flag = FLAG
        self.is_cuda_available()
        self.reward = Reward(tag=reward)

    def is_cuda_available(self):
        '''
            is_cuda_available verifica si esta disponible el gpu,
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
            is_contained verifica si los valores de (x, y, z) estan dentro
            de los rangos definidos por el ambiente;

            state: vector de 12 o 18 posiciones;

            regresa un valor booleano.
        '''
        x = state[3:6]
        high = self.observation_space.high[3:6]
        low = self.observation_space.low[3:6]
        aux = np.logical_and(low <= x, x <= high)
        return aux.all()

    def is_stable(self, state):
        '''
            is_stable verifica el drone esta estable, cerca del objetivo;

            state: vector de 12 o 18 posiciones;

            regresa un valor booleano.
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
            get_score verifica el drone esta contenido y estable;

            state: vector de 12 o 18 posiciones;

            regresa dos puntajes.
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
            get_reward calcula el reward dado el estado del drone;

            state: vector de 12 o 18 posiciones;

            regresa valor real.

        x = state[3]
        x_ = 1  # * np.ones(3)
        r = 0.0
        vel = np.concatenate([state[0:3], state[6:9]])
        x_st = np.logical_and(- x_ <= x, x <= x_)
        if x_st.all():
            r = 1.1
        d1 = norm(x)
        d2 = norm(vel)
        d3 = norm(np.identity(3) - rotation_matrix(state[9:]))
        '''
        return self.reward.eval(state, action)

    def is_done(self):
        '''
            is_done verifica si el drone ya termino de hacer su tarea;

            regresa valor booleano.
        '''
        if self.i == self.steps-2:  # Si se te acabo el tiempo
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
            step realiza la interaccion entre el agente y el ambiente en
            un paso de tiempo;

            action: arreglo de 4 posiciones con valores entre [low, high];

            regresa la tupla (a, r, ns, d)
        '''
        w1, w2, w3, w4 = action + W0
        t = [self.time[self.i], self.time[self.i+1]]
        y_dot = odeint(self.f, self.state, t, args=(w1, w2, w3, w4))[
            1]  # , Dfun=self.jac)[1]
        self.state = y_dot
        reward = self.get_reward(y_dot, action)
        done = self.is_done()
        self.i += 1
        return action, reward, self.state, done

    def reset(self):
        '''
            reset fija la condición inicial para cada simulación del drone

            regresa el estado actual del drone.
        '''
        self.i = 0
        self.state = self.observation_space.sample()
        return self.state

    def set_time(self, steps, time_max):
        '''
            set_time fija la cantidad de pasos y el tiempo de simulación;

            steps: candidad de pasos (int);
            time_max: tiempo de simulación (float).
        '''
        self.time_max = time_max
        self.steps = steps
        self.time = np.linspace(0, self.time_max, self.steps)

    def render(self, mode='human', close=False):
        '''
            render hace una simulación visual del drone.
        '''
        pass


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def __init__(self, env):
        super().__init__(env)
        self.noise = OUNoise(env.action_space)
        self.noise_on = True

    def action(self, action):
        '''
            action transforma una accion de valores entre [-1, 1] a
            los valores [low, high];

            action: arreglo de 4 posiciones con valores entre [-1, 1];

            regresa una acción entre [low, high].
        '''
        high = self.action_space.high
        low = self.action_space.low
        act_k = (high - low) / 2.
        act_b = (high + low) / 2.
        action = act_k * action + act_b
        if self.noise_on:
            action = self.noise.get_action(action, self.i)

        return action

    def reverse_action(self, action):
        '''
            reverse_action transforma una accion entre los valores low y high
            del ambiente a valores entre [-1, 1];

            action: arreglo de 4 posiciones con valores entre [low, high];

            regresa una acción entre [-1, 1].
        '''
        high = self.action_space.high
        low = self.action_space.low
        act_k_inv = 2./(high - low)
        act_b = (high + low) / 2.
        return act_k_inv * (action - act_b)


class AgentEnv(NormalizedEnv, gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate((self.observation_space.low[:9], - np.ones(9)))
        high = np.concatenate((self.observation_space.high[:9], np.ones(9)))
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        '''
            observation transforma un estado de R^12 a un estado de
            R^18;

            obs: estado de R^12;

            regresa un estado en R^18.
        '''
        angles = obs[9:]
        ori = np.matrix.flatten(rotation_matrix(angles))
        return np.concatenate([obs[0:9], ori])

    def reverse_observation(self, obs):
        '''
            reverse_observation transforma un estado de R^18 a un estado de
            R^12;

            obs: estado en R^18;

            regresa un estado en R^12.
        '''
        mat = obs[9:].reshape((3, 3))
        psi = np.arctan(mat[1, 0]/mat[0, 0])
        theta = np.arctan(- mat[2, 0]/np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        phi = np.arctan(mat[2, 1] / mat[2, 2])
        angles = np.array([psi, theta, phi])
        return np.concatenate([obs[0:9], angles])

    def step(self, a):
        '''
            step modifica el método original del super;

            a: arreglo de 4 posiciones con valores entre [-1, 1];

            regresa la tupla (a, r, ns, d).
        '''
        action, reward, state, done = super().step(a)
        state = self.observation(state)
        action = self.reverse_action(action)
        return action, reward, state, done

    def reset(self):
        '''
            reset modifica el método del super

            regresa el estado actual del drone en 18 posiciones.
        '''
        state = super().reset()
        return self.observation(state)
