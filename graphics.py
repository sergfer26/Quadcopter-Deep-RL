
from matplotlib import pyplot as plt
import numpy as np
from params import PARAMS_TRAIN

SHOW = PARAMS_TRAIN['SHOW']
def reverse_observation(obs):
        '''
            reverse_observation transforma un estado de R^18 a un estado de
            R^12;

            obs: estado en R^18

            regresa un estado en R^12
        '''
        mat = obs[9:].reshape((3, 3))
        psi = np.arctan(mat[1, 0]/mat[0, 0])
        theta = np.arctan(- mat[2, 0]/np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        phi = np.arctan(mat[2, 1] / mat[2, 2])
        angles = np.array([psi, theta, phi])
        return np.concatenate([obs[0:9], angles])



def nsim2D(n, agent, env, PATH):
    fig, ((w1, w2), (r1, r2), (p1, p2), (q1, q2),(u1,u2),(v1,v2)) = plt.subplots(6, 2)
    t = env.time
    for _ in range(n):
        state = env.reset()
        env.noise.reset()
        X = [env.state]
        while True:
            action = agent.get_action(state)
            #action = get_action(env.state, env.goal)
            _, reward, new_state, done = env.step(action)
            X.append(env.state)
            if done:
                break
        X = np.array(X)
        u = X[:, 0]; v= X[:, 2]; w = X[:, 2]
        x = X[:, 3]; y = X[:, 4]; z = X[:, 5]
        p = X[:, 6]; q = X[:, 7]; r = X[:, 8]
        psi = X[:, 9]; theta = X[:, 10]; phi = X[:, 11]
        cero = np.zeros(len(z))
        w1.plot(t, z, c='b',alpha =0.3)
        w1.set_ylabel('z')
        w2.plot(t, w, c='b',alpha =0.3)
        w2.set_ylabel('dz')
        
        r1.plot(t, psi, c='r',alpha =0.3)
        r1.set_ylabel('$\psi$')
        r2.plot(t, r, c='r',alpha =0.3)
        r2.set_ylabel('d$\psi$')

        p1.plot(t, phi, c='orange',alpha =0.3)
        p1.set_ylabel('$\phi$')
        p2.plot(t, p, c='orange',alpha =0.3)
        p2.set_ylabel(' d$\phi$')

        q1.plot(t, theta,c = 'green',alpha =0.3)
        q1.set_ylabel('$ \\theta$')
        q2.plot(t, q,c = 'green',alpha =0.3)
        q2.set_ylabel(' d$ \\theta$')
        #q2.set_ylim(0.01,-0.01)

        u1.plot(t, x,c = 'skyblue',alpha =0.3)
        u1.set_ylabel('x')
        u2.plot(t,u,c = 'skyblue',alpha =0.3)
        u2.set_ylabel('dx')

        v1.plot(t, y,c  ='mediumpurple',alpha =0.3)
        v1.set_ylabel('y')
        v2.plot(t,v,c  ='mediumpurple',alpha =0.3)
        v2.set_ylabel('dy')

        w1.plot(t, cero, '--', c='k', alpha=0.5)
        w2.plot(t, cero, '--', c='k', alpha=0.5)
        r2.plot(t, cero, '--', c='k', alpha=0.5)
        p2.plot(t, cero, '--', c='k', alpha=0.5)
        q2.plot(t, cero, '--', c='k', alpha=0.5)
    fig.suptitle(str(n) + ' Vuelos')

    if SHOW:
        plt.show()
    else:
        fig.set_size_inches(33., 21.)
        plt.savefig(PATH + '/vuelos_2D.png', dpi=300)

def nsim3D(n, agent, env, PATH):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    mean_episode_reward = 0.0
    for _ in range(n):
        state = env.reset()
        env.noise.reset()
        x0, y0, z0 = env.state[3:6]
        X, Y, Z = [x0], [y0], [z0]
        while True:
            action = agent.get_action(state)
            #action = get_action(env.state, env.goal)
            _, reward, new_state, done = env.step(action)
            x, y, z = env.state[3:6]
            Z.append(z)
            X.append(x)
            Y.append(y)
            state = new_state
            mean_episode_reward += reward
            if done:
                break
        ax.plot(x0, y0, z0, '.r', markersize=15)
        ax.plot(X, Y, Z, '.b', alpha=0.5, markersize=5)
    fig.suptitle(r'$\overline{Cr}_t = $' +
                 '{} '.format(mean_episode_reward/n), fontsize=20)
    ax.plot(0, 0, 0, '.r', alpha=1, markersize=1)
    if SHOW:
        plt.show()
    else:
        fig.set_size_inches(33., 21.)
        plt.savefig(PATH + '/vuelos.png', dpi=300)