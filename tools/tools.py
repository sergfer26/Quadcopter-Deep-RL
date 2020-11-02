import numpy as np
from matplotlib import pyplot as plt
# import plotly.graph_objects as go


def sub_plot_state(t, x, y, z, axis, axis_labels, labels =[None, None, None], c=['k', 'k', 'k'], alpha=1):
    axis[0].plot(t, x, c=c[0], label=labels[0], alpha=alpha)
    axis[0].set_ylabel(axis_labels[0])
    axis[1].plot(t, y, c=c[1], label=labels[1], alpha=alpha)
    axis[1].set_ylabel(axis_labels[1])
    axis[2].plot(t, z, c=c[2], label=labels[2], alpha=alpha)
    axis[1].set_ylabel(axis_labels[2])
    if labels[0]:
        axis[0].legend()
    if labels[1]:
        axis[1].legend()
    if labels[2]:
        axis[2].legend()
    

def imagen2d(X, t,show=True, path=None):
    u, v, w, x, y, z, p, q, r, psi, theta, phi = X
    # _, (U, V, W, R, P, Q) = plt.subplots(6, 2)
    fig, (X, dX, Phi, dPhi) = plt.subplots(4, 3)
    cero = np.zeros(len(z))

    labels = (str(round(x[-1], 4)), str(round(y[-1], 4)), str(round(z[-1], 4)))
    sub_plot_state(t, x, y, z, X, axis_labels=['x', 'y', 'z'], c=['y', 'c', 'b'], labels=labels)

    labels = (str(round(u[-1], 4)), str(round(v[-1], 4)), str(round(w[-1], 4)))
    sub_plot_state(t, u, v, w, dX, axis_labels=['dx', 'dy', 'dz'], c=['y', 'c', 'b'], labels=labels)

    labels = (str(round(psi[-1], 4)), str(round(theta[-1], 4)), str(round(phi[-1], 4)))
    sub_plot_state(t, psi, theta, phi, Phi, axis_labels=['$\psi$', '$\\theta$', '$\phi$'], c=['r', 'k', 'g'], labels=labels)

    labels = (str(round(r[-1], 4)), str(round(q[-1], 4)), str(round(p[-1], 4)))
    sub_plot_state(t, r, q, p, dPhi, axis_labels=['$d\psi$', '$d\\theta$', '$d\phi$'], c=['r', 'k', 'g'], labels=labels)

    X[0].plot(t, cero + 15, '--', c='k', alpha=0.5)
    X[1].plot(t, cero + 15, '--', c='k', alpha=0.5)
    X[2].plot(t, cero + 15, '--', c='k', alpha=0.5)

    dX[0].plot(t, cero, '--', c='k', alpha=0.5)
    dX[1].plot(t, cero, '--', c='k', alpha=0.5)
    dX[2].plot(t, cero, '--', c='k', alpha=0.5)

    Phi[0].plot(t, cero, '--', c='k', alpha=0.5)
    Phi[1].plot(t, cero, '--', c='k', alpha=0.5)
    Phi[2].plot(t, cero, '--', c='k', alpha=0.5)

    dPhi[0].plot(t, cero, '--', c='k', alpha=0.5)
    dPhi[1].plot(t, cero, '--', c='k', alpha=0.5)
    dPhi[2].plot(t, cero, '--', c='k', alpha=0.5)

    if show:
        plt.show()
    else:
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(path + '/sim.png' , dpi=300)
        plt.close()


def imagen_action(action, t, show=True, path=None):
    _, ax = plt.subplots(4, 1)
    #cero = np.zeros(len(action[0])) 
    action = np.array(action)
    ax[0].plot(t, action[:, 0], c='b')
    ax[0].set_ylabel('$a_1$')
    ax[0].set_ylim([-5, 5])

    ax[1].plot(t, action[:, 1], c='r')
    ax[1].set_ylabel('$a_2$')
    ax[1].set_ylim([-5, 5])

    ax[2].plot(t, action[:, 2], c='g')
    ax[2].set_ylabel('$a_3$')
    ax[2].set_ylim([-5, 5])

    ax[3].plot(t, action[:, 3])
    ax[3].set_ylabel('$a_4$')
    ax[3].set_ylim([-5, 5])

    if show:
        plt.show()
    else: 
        plt.savefig(path + '/actions.png', dpi=300)
        plt.close()

'''
def imagen(X, Y, Z):
    fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=1, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
'''

def hist(z,w, env):
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.hist(z,label = 'Z', color = 'navy')
    ax2.hist(w,label = 'W')
    fig.suptitle(' beta  = ' + str(env.beta) + ', ' +'epsilon = ' + str(env.epsilon) , fontsize=16)
    ax1.legend()
    ax2.legend()
    plt.show()
