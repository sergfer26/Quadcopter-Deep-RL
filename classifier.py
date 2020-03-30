#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy import pi
from step import simulador


C = 10 ** -3


def random_position(Ym, Ysd):
    '''
    Genera un arreglo de normales donde cada elemento i tiene
    media mean_y[i] y varianza sd_y[i].

    param mean_y: arreglo con medias
    param sd_y: arreglo de varianzas

    regresa: arreglo de normales
    '''
    g = lambda x, y: normal(x, y)
    return g(Ym, Ysd)


def criterio(z0, zT, ze):
    '''
    z_0 es el valor inicial de la variable,z_50 es el valor final
    y z_e es el valor al que se debe estabilizar
    '''
    #print(z_50,z_e ,np.abs(z_50-z_e))
    if abs(zT - ze) < C:
        return True
    else:
        return False


def test(Y0, Ze):
    '''
    y0 es la condicion inicial y  Z_e es el punto en donde quiero estabilizar
    '''
    z_e, psi_e, phi_e, theta_e = Ze
    _, _, w_0, p_0, q_0, r_0, psi_0, theta_0, phi_0, _, _, z_0 = Y0
    y_f = simulador(Y0, Ze, 50, 1500)[-1, :]
    _, _, w_f, p_f, q_f, r_f, psi_f, theta_f, phi_f, _, _, z_f = y_f
    z = criterio(z_0, z_f, z_e) and criterio(w_0, w_f, 0)
    psi = criterio(psi_0, psi_f, psi_e) and criterio(r_0, r_f, 0)
    phi = criterio(phi_0, phi_f, phi_e) and criterio(p_0, p_f, 0)
    theta = criterio(theta_0, theta_f, theta_e) and criterio(q_0, q_f, 0)
    #tests = list(map(int, [z, psi, phi, theta]))
    return z and psi and phi and theta


def postion_vs_velocity(z, w, psi, r, phi, p, theta, q, cluster):
    '''
    Grafica la posición vs la velocidad
    '''
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


def n_tests(Ze, n):
    Ym = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
    Ysd = np.ones(12)
    cluster = np.zeros(n)
    X = np.zeros([n, 12])
    for i in range(n):
        Y = random_position(Ym, Ysd)
        clase = test(Y, Ze)
        X[i, ] = Y
        if clase:
            cluster[i] = 1
    
    z = X[:, 11]
    w = X[:, 2]
    psi = X[:, 6]
    r = X[:, 5]
    phi = X[:, 8]
    p = X[:, 3]
    theta = X[:, 7]
    q = X[:, 4]
    postion_vs_velocity(z, w, psi, r, phi, p, theta, q, cluster)


if __name__ == "__main__":
    Y0 = np.array([0, 0, 0, 0, 0, 0, pi/20, pi/20, pi/20, 0, 0, 15])
    Ze = (10, 0, 0, 0)
    n_tests(Ze, 100)
