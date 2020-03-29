#!/usr/bin/env python3
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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


def escribe(X, Y, Z, phi, theta, psi):
    posicion = open('XYZ.txt','w')
    angulos = open('ang.txt','w')
    np.savetxt('XYZ.txt', [X, Y, Z])
    np.savetxt('ang.txt',[phi, psi, theta])
    posicion.close()
    angulos.close()


def imagen(X, Y, Z):
    fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=1, colorscale='Viridis', opacity=0.8))])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def imagen2d(z, w, psi, r, phi, p, theta, q, t):
    f, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
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

    w2.plot(t, cero, '--', c='k', alpha=0.5)
    r2.plot(t, cero, '--', c='k', alpha=0.5)
    p2.plot(t, cero, '--', c='k', alpha=0.5)
    q2.plot(t, cero, '--', c='k', alpha=0.5)

    plt.show()


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
    pass
    # w1, w2, w3, w4 = (0, 53.6666, 55.6666, 1)
    # t = np.linspace(0, 10, 1000)
    # y = 0, 0, 0, 0, 0, 0, 2, 3, 10, 0, 0, 10
    # W = [w1, w2, w3, w4]
    # sol = odeint(f, y, t, args=(w1, w2, w3, w4))

    #psi = sol[:,6]
    #theta = sol[:,7]
    #phi = sol[:,8]
    # X = sol[:,9]
    # Y = sol[:,10]
    # Z = sol[:,11]

    #escribe()
    #imagen(X, Y, Z)
