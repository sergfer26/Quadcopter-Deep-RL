#!/usr/bin/env python3
# from cmath import cos
import pandas as pd
import numpy as np
from scipy.integrate import odeint
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from numpy.linalg import norm
from Linear.constants import CONSTANTS
from scipy.spatial.transform import Rotation as R

G = CONSTANTS['G']
Ixx = CONSTANTS['Ixx']
Iyy = CONSTANTS['Iyy']
Izz = CONSTANTS['Izz']
B = CONSTANTS['B']
M = CONSTANTS['M']
L = CONSTANTS['L']
K = CONSTANTS['K']

'''
G = 9.81
Ixx, Iyy, Izz = 1, 1, 0.5  # (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140 * 1e-7, 1, 0.225
K = 2.980 * 1e-6  # kt
'''

omega_0 = np.sqrt((G * M)/(4 * K))
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0


def sec(x): return 1/np.cos(x)


def angles2rotation(angles, flatten=True):
    z, y, x = angles  # psi, theta, phi
    r = R.from_euler('xyz', [x, y, z], degrees=False)
    r = r.as_matrix()
    if flatten:
        r = r.flatten()
    return r


def rotation2angles(rot):
    if rot.shape[0] == 9:
        rot = rot.reshape((3, 3))
    r = R.from_matrix(rot)
    return r.as_euler('zyx', degrees=False)


def f(X, t, w1, w2, w3, w4):  # Sistema dinámico
    '''
        f calcula el vector dot_x = f(x, t, w) (sistema dinamico);

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regresa dot_x
    '''
    u, v, w, _, _, _, p, q, r, _, theta, phi = X
    # Ixx, Iyy, Izz = I
    W = np.array([w1, w2, w3, w4])
    du = r * v - q * w - G * np.sin(theta)
    dv = p * w - r * u - G * np.cos(theta) * np.sin(phi)
    dw = q * u - p * v + G * np.cos(phi) * np.cos(theta) - (K/M) * norm(W) ** 2
    dp = ((L * B) / Ixx) * (w4 ** 2 - w2 ** 2) - q * r * ((Izz - Iyy) / Ixx)
    dq = ((L * B) / Iyy) * (w3 ** 2 - w1 ** 2) - p * r * ((Ixx - Izz) / Iyy)
    dr = (B/Izz) * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
    dpsi = (q * np.sin(phi) + r * np.cos(phi)) * (1 / np.cos(theta))
    dtheta = q * np.cos(phi) - r * np.sin(phi)
    dphi = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    dx = u
    dy = v
    dz = w
    return np.array([du, dv, dw, dx, dy, dz, dp, dq, dr, dpsi, dtheta, dphi])


def f_x(state, actions):
    '''
        jac_f calcula el jacobiano de la funcion f;

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regrasa la matriz J.
    '''
    u, v, w, _, _, _, p, q, r, _, theta, phi = state
    a1 = (Izz - Iyy)/Ixx
    a2 = (Ixx - Izz)/Iyy
    J = np.zeros((12, 12))

    J[0, :] = np.array(
        [0, r, -q, 0, 0, 0, 0, -w, v, 0, - G * np.cos(theta), 0])
    J[1, :] = np.array([-r, 0, p, 0, 0, 0, w, 0, -u, 0, G *
                        np.sin(theta) * np.sin(phi),
                        -G * np.cos(theta) * np.cos(phi)])
    J[2, :] = np.array([q, -p, 0, 0, 0, 0, -v, u, 0, 0, G *
                        np.sin(theta) * np.cos(phi),
                        -G * np.cos(theta) * np.sin(phi)])

    J[3, 0] = 1.
    J[4, 1] = 1.
    J[5, 2] = 1.
    J[6, 7] = -r * a1
    J[6, 8] = -q * a1
    J[7, 6] = -r * a2
    J[7, 8] = -p * a2
    # ddr = np.zeros(12) # J[8, :]

    J[9, 7: 9] = np.array([np.sin(phi), np.cos(phi) * sec(theta)])
    J[9, 10:] = np.array([r * np.cos(phi) * np.tan(theta) * sec(theta),
                          (q * np.cos(phi) - r * np.sin(phi)) * sec(theta)])

    J[10, 7: 9] = np.array([np.cos(phi), - np.sin(phi)])
    J[10, -1] = -q * np.sin(phi) - r * np.cos(phi)

    J[11, 6: 9] = np.array(
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)])
    J[11, 10:] = np.array([(q * np.sin(phi) + r * np.cos(phi)) * sec(theta) **
                           2, (q * np.cos(phi) - r * np.sin(phi)) * np.tan(theta)])

    return J


def jac_f(X, t, w1, w2, w3, w4):
    '''
        jac_f calcula el jacobiano de la funcion f;

        X: es un vector de 12 posiciones;
        t: es un intervalo de tiempo [t1, t2];
        wi: es un parametro de control, i = 1, 2, 3, 4;

        regrasa la matriz J.
    '''
    u, v, w, _, _, _, p, q, r, _, theta, phi = X
    a1 = (Izz - Iyy)/Ixx
    a2 = (Ixx - Izz)/Iyy
    J = np.zeros((12, 12))
    ddu = np.zeros(12)
    J[0, 1:3] = [r, -q]
    ddv = [0, r, -q, 0, 0, 0, w, 0, -u, G *
           np.sin(theta) * np.sin(phi), -G * np.cos(theta) * np.cos(phi)]
    ddw = [q, -p, 0, 0, 0, 0, -v, u, 0, 0, G *
           np.sin(theta) * np.cos(phi), -G * np.cos(theta) * np.sin(phi)]
    ddx = np.zeros(12)
    ddx[0] = 1
    ddy = np.zeros(12)
    ddx[1] = 1
    ddz = np.zeros(12)
    ddx[2] = 1
    ddp = np.zeros(12)
    ddp[7] = -r * a1
    ddp[8] = -q * a1
    ddq = np.zeros(12)
    ddq[6] = -r * a2
    ddq[8] = -p * a2
    ddr = np.zeros(12)

    ddpsi = np.zeros(12)
    ddpsi[7: 9] = [np.sin(phi), np.cos(phi) * sec(theta)]
    ddpsi[10:] = [r * np.cos(phi) * np.tan(theta) * sec(theta),
                  (q * np.cos(phi) - r * np.sin(phi)) * sec(theta)]

    ddtheta = np.zeros(12)
    ddtheta[7: 9] = [np.cos(phi), - np.sin(phi)]
    ddtheta[-1] = -q * np.sin(phi) - r * np.cos(phi)

    ddphi = np.zeros(12)
    ddphi[6: 9] = [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)]
    ddphi[10:] = [(q * np.sin(phi) + r * np.cos(phi)) * sec(theta) **
                  2, (q * np.cos(phi) - r * np.sin(phi)) * np.tan(theta)]

    J = np.array([ddu, ddv, ddw, ddx, ddy, ddz, ddp,
                 ddq, ddr, ddpsi, ddtheta, ddphi])
    return J


def f_u(state, actions):
    w1, w2, w3, w4 = actions
    J = np.zeros((12, 4))
    J[2, :] = -2 * (K/M) * actions
    J[3, :] = 2 * (L/Ixx) * np.array([0, -w2, 0, w4])
    J[4, :] = 2 * (L/Iyy) * np.array([-w1, 0, w3, 0])
    J[5, :] = 2 * (B/Izz) * actions
    return J


def escribe(X, Y, Z, phi, theta, psi):
    posicion = open('XYZ.txt', 'w')
    angulos = open('ang.txt', 'w')
    np.savetxt('XYZ.txt', [X, Y, Z])
    np.savetxt('ang.txt', [phi, psi, theta])
    posicion.close()
    angulos.close()


def imagen(X, Y, Z):
    fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers',
                                       marker=dict(size=1,
                                                   colorscale='Viridis',
                                                   opacity=0.8))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def imagen2d(z, w, psi, r, phi, p, theta, q, t, show=True, dir_=None):
    fig, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
    cero = np.zeros(len(z))

    w1.plot(t, z, c='b')
    w1.set_ylabel('z')
    w2.plot(t, w, c='b')
    w2.set_ylabel('dz')

    r1.plot(t, psi, c='r')
    r1.set_ylabel('$\psi$')
    r2.plot(t, r, c='r')
    r2.set_ylabel('d$\psi$')

    p1.plot(t, phi, c='g')
    p1.set_ylabel('$\phi$')
    p2.plot(t, p, c='r')
    p2.set_ylabel(' d$\phi$')

    q1.plot(t, theta)
    q1.set_ylabel('$ \\theta$')
    q2.plot(t, q)
    q2.set_ylabel(' d$ \\theta$')
    q2.set_ylim(0.001, -0.001)

    w1.plot(t, cero + 15, '--', c='k', alpha=0.5)
    w2.plot(t, cero, '--', c='k', alpha=0.5)
    r2.plot(t, cero, '--', c='k', alpha=0.5)
    p2.plot(t, cero, '--', c='k', alpha=0.5)
    q2.plot(t, cero, '--', c='k', alpha=0.5)
    if show:
        plt.show()
    else:
        fig.set_size_inches(33., 21.)
        plt.savefig(dir_+'/sim.png')


def postion_vs_velocity(z, w, psi, r, phi, p, theta, q, cluster):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Sharing x per column, y per row')

    ax1.scatter(z, w, c=cluster, s=50)
    ax1.set_xlabel('z')
    ax1.set_ylabel('w')

    ax2.plot(psi, r, c=cluster, s=50)
    ax2.set_xlabel('$\psi$')
    ax2.set_ylabel('r')

    ax3.plot(phi, p, c=cluster, s=50)
    ax3.set_xlabel('$\phi')
    ax3.set_ylabel('p')

    ax4.plot(theta, q, c=cluster, s=50)
    ax3.set_xlabel('$\\theta')
    ax3.set_ylabel('q')


if __name__ == "__main__":
    w1, w2, w3, w4 = (0, 53.6666, 55.6666, 1)
    t = np.linspace(0, 10, 2600)
    y = 0, 0, 0, 0, 0, 0, 2, 3, 10, 0, 0, 10
    W = [w1, w2, w3, w4]
    sol = odeint(f, y, t, args=(w1, w2, w3, w4))

    psi = sol[:, 6]
    theta = sol[:, 7]
    phi = sol[:, 8]
    X = sol[:, 9]
    Y = sol[:, 10]
    Z = sol[:, 11]

    # escribe()
    imagen(X, Y, Z)
