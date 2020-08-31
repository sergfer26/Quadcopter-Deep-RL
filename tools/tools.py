import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go


def reset_time(env, tamaño, tiempo_max):
    env.time_max = tiempo_max
    env.tam = tamaño
    env.time = np.linspace(0, env.time_max, env.tam)

def get_score(state, env):
    z = state[5]
    w = state[2]
    score1 = 0
    score2 = 0
    if env.goal[5] - 0.20 < z < env.goal[5] + 0.20 and abs(w) < 0.5:
        score1 = 1
    if env.observation_space.low[5] < z < env.observation_space.high[5]:
        score2 = 1
    return score1, score2

def imagen2d(z, w, psi, r, phi, p, theta, q, t, show=True):
    fig, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
    cero = np.zeros(len(z))

    w1.plot(t, z, c='b',label = str(round(z[-1],4)))
    w1.set_ylabel('z')
    w2.plot(t, w, c='b',label = str(round(w[-1],4)))
    w2.set_ylabel('dz')
    w1.legend()
    w2.legend()

    r1.plot(t, psi, c='r')
    r1.set_ylabel('$\psi$')
    r2.plot(t, r, c='r')
    r2.set_ylabel('d$\psi$')

    p1.plot(t, phi, c='g')
    p1.set_ylabel('$\phi$')
    p2.plot(t, p, c='g')
    p2.set_ylabel(' d$\phi$')

    q1.plot(t, theta)
    q1.set_ylabel('$ \\theta$')
    q2.plot(t, q)
    q2.set_ylabel(' d$ \\theta$')

    w1.plot(t, cero + 15, '--', c='k', alpha=0.5)
    w2.plot(t, cero, '--', c='k', alpha=0.5)
    r2.plot(t, cero, '--', c='k', alpha=0.5)
    p2.plot(t, cero, '--', c='k', alpha=0.5)
    q2.plot(t, cero, '--', c='k', alpha=0.5)

    if show:
        plt.show()
    else: 
        fig.save


def imagen_action(action, t):
    _, ax = plt.subplots(4, 1)
    cero = np.zeros(len(action[0])) 

    ax[0].plot(t, action[0], c='b')
    ax[0].set_ylabel('$a_1$')

    ax[1].plot(t, action[1], c='r')
    ax[1].set_ylabel('$a_2$')

    ax[2].plot(t, action[2], c='g')
    ax[2].set_ylabel('$a_3$')

    ax[3].plot(t, action[3])
    ax[3].set_ylabel('$a_4$')

    ax[0].plot(t, cero + 15, '--', c='k', alpha=0.5)
    ax[1].plot(t, cero + 15, '--', c='k', alpha=0.5)
    ax[2].plot(t, cero + 15, '--', c='k', alpha=0.5)
    ax[3].plot(t, cero + 15, '--', c='k', alpha=0.5)

    plt.show()





def imagen(X, Y, Z):
    fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=1, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def hist(z,w):
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.hist(z,label = 'Z', color = 'navy')
    ax2.hist(w,label = 'W')
    fig.suptitle(' beta  = ' + str(env.beta) + ', ' +'epsilon = ' + str(env.epsilon) , fontsize=16)
    ax1.legend()
    ax2.legend()
    plt.show()
