# from statistics import covariance
import numpy as np
import torch
from numpy.linalg import inv
from numpy.linalg import norm as norma
# from gmr import covariance_initialization, kmeansplusplus_initialization
from gmr.sklearn import GaussianMixtureRegressor
from gmr import MVN
from Linear.agent import LinearAgent
from env import QuadcopterEnv, ObsEnv
from GCL.taylor import jacobian, hessian
from torch import cos, sin
from utils import plot_state
from matplotlib import pyplot as plt

from progressbar import progressbar
# agent = LinearAgent(env, 100000)
C1 = 0.5
C2 = 1.0
C3 = 0.005
eps = 1e-4


def observation(obs):
    '''
        observation transforma un estado de R^12 a un estado de
        R^18;
        obs: estado de R^12;
        regresa un estado en R^18.
    '''
    angles = obs[9:]
    ori = np.matrix.flatten(rotation_matrix(angles).numpy())
    return np.concatenate([obs[0:9], ori])


def reverse_observation(obs):
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


def rotation_matrix(angles):
    '''
        rotation_matrix obtine la matriz de rotación de los angulos de Euler
        https://en.wikipedia.org/wiki/Rotation_matrix;

        angles: son los angulos de Euler psi, theta y phi con respecto a los
        ejes x, y, z;

        regresa: la matriz R.
    '''
    z, y, x = angles  # psi, theta, phi
    R = torch.tensor([
        [cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x),
         cos(z) * sin(y) * cos(x) + sin(z) * sin(x)],
        [sin(z) * cos(y), sin(z) * cos(y) * sin(x) + cos(z) * cos(x),
         sin(z) * sin(y) * cos(x) - cos(z) * sin(x)],
        [- sin(y), cos(y) * np.sin(x), cos(y) * cos(x)]
    ])
    return R


def get_reward(z):
    state, action = z[0:18], z[18:]
    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    penalty = 0.5 * torch.norm(state[3:6])
    mat = state[9:].reshape((3, 3))
    penalty += 1.0 * torch.norm(torch.eye(3) - mat)
    penalty += 0.005 * torch.norm(action)
    return - penalty


class LGC(LinearAgent):
    '''
    Linear Gaussian Controllers
    '''

    def __init__(self, env, max_memory_size, n_components):
        super().__init__(env, max_memory_size)
        # p(u_t| x_t) = N(K_t * x + k_t, C_t)
        self.k = [np.zeros((self.num_actions, 1)) for _ in range(0, env.steps)]
        self.K = [np.zeros((self.num_actions, self.num_states))
                  for _ in range(0, env.steps)]
        self.C = [np.ones((self.num_actions, self.num_actions))
                  for _ in range(0, env.steps)]
        self.gmr = GaussianMixtureRegressor(n_components=n_components)
        self.env = env

    def fit_controlller(self, traj):
        # backward pass
        u_dim = self.num_actions
        x_dim = self.num_states
        states, _, _, dones = traj
        dones[-1] = [1.]
        n, _ = states.shape
        for e in range(n-1, -1, -1):
            Q_xu = self.Qxu(traj, e)
            Q_xuxu = self.Qxuxu(traj, e)
            Q_uu = Q_xuxu[x_dim:x_dim + u_dim, x_dim: x_dim + u_dim]
            Q_u = Q_xu[x_dim: x_dim + u_dim]
            Q_ux = Q_xuxu[x_dim:, :x_dim]
            Q_uu_1 = inv(Q_uu + eps * np.identity(u_dim))
            self.K[e] = - np.dot(Q_uu_1, Q_ux)
            self.k[e] = - np.dot(Q_uu_1, Q_u)
            self.C[e] = Q_uu_1

    def V(self, traj, idx):
        states, _, _, _ = traj
        x = states[idx, :]
        V_x, V_xx = self.Vx(traj, idx), self.Vxx(traj, idx)
        return (x.T * V_x * x)/2 + x.T * V_xx + eps

    def Vx(self, traj, idx):
        u_dim = self.num_actions
        x_dim = self.num_states
        Q_xu = self.Qxu(traj, idx)
        Q_xuxu = self.Qxuxu(traj, idx)
        Q_uu = Q_xuxu[x_dim:x_dim + u_dim, x_dim: x_dim + u_dim]
        Q_x, Q_u = Q_xu[0:x_dim], Q_xu[x_dim: x_dim + u_dim]
        Q_ux = Q_xuxu[x_dim:, :x_dim]
        Q_uu_1 = inv(Q_uu + eps * np.identity(u_dim))
        return Q_x - np.dot(Q_ux.T, np.dot(Q_uu_1, Q_u))

    def Vxx(self, traj, idx):
        u_dim = self.num_actions
        x_dim = self.num_states
        Q_xuxu = self.Qxuxu(traj, idx)
        Q_uu = Q_xuxu[x_dim:x_dim + u_dim, x_dim: x_dim + u_dim]
        Q_xx = Q_xuxu[0:x_dim, 0:x_dim]
        Q_ux = Q_xuxu[x_dim:, :x_dim]
        Q_uu_1 = inv(Q_uu + eps * np.identity(u_dim))
        return Q_xx - np.dot(Q_ux.T, np.dot(Q_uu_1, Q_ux))

    def Q(self, traj, idx):
        states, actions, new_states, _ = traj
        x, u, _ = states[idx, :], actions[idx,
                                          :], new_states[idx, :]
        y = np.concatenate([x, u], axis=-1)
        Q_xuxu = self.Qxuxu(traj, idx)
        Q_xu = self.Qxu(traj, idx)
        # Q_xu, Q_xuxu = self.dQ(x, u, nx, second_order=True)
        return np.dot(y.T, np.dot(Q_xuxu, y))/2 + np.dot(y.T, Q_xu) + eps

    def Qxu(self, traj, idx):
        states, actions, _, dones = traj
        x, u, done = states[idx, :], actions[idx,
                                             :], dones[idx]
        x_t = torch.tensor(x)
        u_t = torch.tensor(u)
        z = torch.cat([x_t, u_t])
        z.requires_grad = True
        ell = get_reward(z)
        ell_xu = jacobian(ell, z).detach().cpu().numpy()
        Q_xu = ell_xu
        if not done.all():
            f = self.fxu(traj, idx)
            V_x = self.Vx(traj, idx + 1)
            Q_xu += np.dot(f.T, V_x)
        return Q_xu

    def Qxuxu(self, traj, idx):
        '''Hessian of function Q'''
        states, actions, new_states, dones = traj
        x, u, done = states[idx, :], actions[idx,
                                             :], dones[idx]
        x_t = torch.tensor(x)
        u_t = torch.tensor(u)
        z = torch.cat([x_t, u_t])
        z.requires_grad = True
        ell = get_reward(z)
        ell_xuxu = hessian(ell, z).detach().cpu().numpy()
        Q_xuxu = ell_xuxu
        if not done.all():
            f_xu = self.fxu(traj, idx)
            V_xx = self.Vxx(traj, idx + 1)
            Q_xuxu += np.dot(f_xu.T, np.dot(V_xx, f_xu))
        return Q_xuxu

    def fxu(self, traj, idx):
        ''' tmp '''
        states, actions, new_states, _ = traj
        x, u, nx = states[idx, :], actions[idx,
                                           :], new_states[idx, :]
        z = np.concatenate([x, u, nx], axis=-1)
        z = z.reshape(1, -1)
        p = self.gmr.gmm_.to_responsibilities(z).flatten()
        e = np.argmax(p)
        f_xu, _ = self.get_posterior_mean(e)
        # fx = fxu[:self.num_states, :self.num_states]
        # fu = fxu[self.num_states:, self.num_states:]
        return f_xu

    def get_posterior_mean(self, idx):
        # f = (f_x, f_u) = mu_y|x
        # mu_y|x = mu_y + sigma_yx * sigma_xx^-1 (x - mu_x)
        mean, cov = self.gmr.gmm_.means[idx], self.gmr.gmm_.covariances[idx]
        mu_y = mean[self.num_states + self.num_actions:]
        mu_x = mean[: self.num_states + self.num_actions]
        sigma_yx = cov[self.num_states + self.num_actions:, :-self.num_states]
        sigma_xx = cov[:self.num_states + self.num_actions,
                       :self.num_states + self.num_actions]
        f_xu = np.dot(sigma_yx, inv(sigma_xx))
        f_t = mu_y + np.dot(sigma_yx, np.dot(inv(sigma_xx), - mu_x))
        return f_xu, f_t

    def get_posterior_cov(self, idx):
        # sigma_y|x = sigma_yy - sigma_yx * sigma_xx^-1 * sigma_xy
        cov = self.gmr.gmm_.covariances[idx]
        sigma_yy = cov[:-self.num_states, :-self.num_states]
        sigma_yx = cov[self.num_states + self.num_actions:, :-self.num_states]
        sigma_xx = cov[self.num_states + self.num_actions:,
                       self.num_states + self.num_actions:]
        sigma_xy = cov[:self.num_states + self.num_actions, :-self.num_states]
        Ft = sigma_yy + np.dot(sigma_yx, np.dot(inv(sigma_xx), sigma_xy))
        return Ft

    def get_action(self, state):
        # n, _ = state.shape
        mean = np.dot(self.K[env.i], state) + self.k[env.i].flatten()
        return MVN(mean=mean, covariance=self.C[env.i]).sample(1)[0]

    def to_controller_density(self, state, action):
        # p(a|s)
        mean = np.dot(self.Kt, state) + self.kt
        p = MVN(mean=mean, covariance=self.Ct).to_probability_density(action)
        return p

    def to_posterior_density(self, state, action, new_state):
        # p(s'|a, s)
        z = np.concatenate([state, action], axis=-1)
        indices = np.arange(0, self.num_states + self.num_actions, 1)
        self.gmr.gmm_.condition(indices, z)
        p = self.gmr.gmm_.condition(
            indices, z).to_probability_density(new_state)
        return p

    def init_density(self, state):
        self.env.observation_space
        pass

    def to_trajectory_density(self, traj):
        # p(s0, a1, s1, ..., s_t, a_{t+1}, s_{t+1}, ...)
        # p(s0) \Pi p(a|s) * p(s'|a, s)
        p = 1
        for state, action, new_state, _ in traj:
            p *= self.to_controller_density(state, action)
            p *= self.to_posterior_density(state, action, new_state)
        return p


def join_trajs(trajs):
    all_states, all_actions = list(), list()
    all_next_states, all_dones = list(), list()
    for traj in trajs:
        states, actions, next_states, dones = traj
        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_dones.append(dones)

    all_states = np.stack(all_states, axis=0)
    all_actions = np.stack(all_actions, axis=0)
    all_next_states = np.stack(all_next_states, axis=0)
    all_dones = np.stack(all_dones, axis=0)
    return all_states, all_actions, all_next_states, all_dones


# if __name__ == "__main__":
n_components = 5
rollouts = 20
env = QuadcopterEnv()
expert = LinearAgent(env, 100000)
agent = LGC(ObsEnv(env), 100000, env.time_max)

for _ in range(rollouts):
    traj = list()
    state = env.reset()
    while True:
        action = expert.get_action(state)
        new_state, reward, done, _ = env.step(action)
        # + normal(0, 1, len(new_state))])
        traj.append([agent.env.observation(state), action,
                    agent.env.observation(new_state), done])
        state = new_state
        if done:
            agent.memory_traj.push(traj)
            break

trajs = agent.memory_traj.sample(rollouts)
train_trajs = trajs[0: int(0.8 * rollouts)]
test_trajs = trajs[int(0.8 * rollouts):]
train_states, train_actions, train_next_states, _ = join_trajs(train_trajs)
test_states, test_actions, test_next_states, _ = join_trajs(test_trajs)
# agent.memory_traj.clear()
X_train = np.concatenate([train_states, train_actions], axis=-1)
X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(train_next_states, axis=0)
agent.gmr.fit(X_train, Y_train)

X_test = np.concatenate([test_states, test_actions], axis=-1)
X_test = np.concatenate(X_test, axis=0)
Y_test = np.concatenate(test_next_states, axis=0)
Y_pred = agent.gmr.predict(X_test)
print(norma(Y_pred - Y_test))
agent.memory_traj.clear()

rollouts = 20
for rollout in progressbar(range(1, rollouts + 1)):
    traj = list()
    state = agent.env.reset()
    while True:
        action = agent.get_action(state)
        new_state, reward, done, _ = agent.env.step(action)
        # + normal(0, 1, len(new_state))])
        traj.append([state, action, new_state, done])
        state = new_state
        if done | agent.env.i == agent.env.steps-2:
            print('ya acabé!')
            agent.memory_traj.push(traj)
            break
        if rollout % 5 == 0:
            trajs = agent.memory_traj.sample(4)
            agent.memory_traj.clear()
            for traj in trajs:
                agent.fit_controlller(traj)

state = agent.env.reset()
names = ['$u$', '$v$', '$w$', '$x$', '$y$', '$z$', '$p$', '$q$', '$r$',
         '$\\psi$', '$\\theta$', '$\\phi$']
dic_state = {name: list() for name in names}
while True:
    action = expert.get_action(state)
    new_state, reward, done, _ = agent.env.step(action)
    # + normal(0, 1, len(new_state))])
    traj.append([state, action, new_state, done])
    state = new_state
    for val, lista in zip(state, dic_state):
        lista.append(val)
    if done:
        break

plot_state(dic_state, t=env.time)
plt.show()
