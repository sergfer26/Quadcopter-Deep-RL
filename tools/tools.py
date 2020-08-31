import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go


def reset_time(env, tamaño, tiempo_max):
    env.time_max = tiempo_max
    env.tam = tamaño
    env.time = np.linspace(0, env.time_max, env.tam)

def get_score(state, env):
    z = state[-1]
    w = state[2]
    if env.goal[-1] - 0.20 < z < env.goal[-1] + 0.20 and abs(w) < 0.25:
        return 1
    else:
        return 0


def imagen2d(z, w, psi, r, phi, p, theta, q, t):
    f, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
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

    plt.show()


def imagen(X, Y, Z):
    fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=1, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    
def imagen_accion():
    t = env.time
    state = funcion(env.reset())
    noise.reset()
    A = []
    while True:
        action = agent.get_action(state)
        action = noise.get_action(action, env.time[env.i])
        control =  action
        A.append(control)
        new_state, reward, done = env.step(control)
        state = new_state
        if done:
            break
    t = env.time[0:len(A)]
    ayuda = W0[0]*np.ones(len(A))
    A = np.array(A)
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, A[:,0], c='b')
    ax[0].set_ylabel('$a_1$')
    ax[1].plot(t, A[:,1], c='r')
    ax[1].set_ylabel('$a_2$')
    ax[2].plot(t, A[:,2], c='g')
    ax[2].set_ylabel('$a_3$')
    ax[3].plot(t, A[:,3])
    ax[3].set_ylabel('$a_4$')
    fig.set_size_inches(30.,18.)
    plt.savefig('testplot.png',dpi=200)
