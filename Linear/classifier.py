#!/usr/bin/env python3
from progress.bar import Bar, ChargingBar
import os, time, random
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, uniform
from numpy import pi
from step import simulador
C = 10 ** -3


def random_position(Ym, perturbations, Ysd=None, normal_=True, dist=None):
    '''
    Genera un arreglo de normales donde cada elemento i tiene
    media mean_y[i] y varianza sd_y[i].

    param mean_y: arreglo con medias
    param sd_y: arreglo de varianzas
    param perturbations: una lista con las posiciones que se 
                         perturbaran

    regresa: arreglo de normales
    '''
    per = perturbations
    if normal_:
        g = lambda x, y: normal(x, y)
        Y = [g(x, y) if p == 1 else 0 for x, y, p in zip(Ym, Ysd, per)] 
    else:
        a = Ym - dist
        b = Ym + dist
        g = lambda x, y: uniform(x,y)
        Y = [g(x, y) if p == 1 else 0 for x, y, p in zip(a, b, per)]
    return np.array(Y)


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

    param z, psi, phi, theta: arreglos de las posiciones en dichas variables
    param w, r, p, q: arreglos de las velocidades en en dichas variables
    cluster: lista con las clases de cada entrada de los arreglos anteriores
    '''
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Sharing x per column, y per row')

    ax1.scatter(z, w, c=cluster, s=10, alpha=0.2)
    ax1.set_xlabel('z')
    ax1.set_ylabel('w')

    ax2.scatter(psi, r, c=cluster, s=10, alpha=0.2)
    ax2.set_xlabel('$\psi$')
    ax2.set_ylabel('r')

    ax3.scatter(phi, p, c=cluster, s=10, alpha=0.2)
    ax3.set_xlabel('$\phi$')
    ax3.set_ylabel('p')

    ax4.scatter(theta, q, c=cluster, s=10, alpha=0.2)
    ax4.set_xlabel('$\\theta$')
    ax4.set_ylabel('q')
    # plt.show()


def n_tests(Ze, n, perturbations, normal_=True, d=None):
    Ym = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10])
    Ysd = np.ones(12)
    Ysd[-1] = 2
    cluster = []
    X = np.zeros([n, 12])
    bar1 = Bar('Procesando:', max=n)
    for i in range(n):
        Y = random_position(Ym, perturbations, Ysd=Ysd, normal_=normal_, dist=d)
        clase = test(Y, Ze)
        X[i, ] = Y
        Y = np.append(Y, clase)
        if clase:
            cluster.append('b')
        else:
            cluster.append('r')
        bar1.next()
        #if i % 100 == 0:
        #    print(i)
    z = X[:, 11]
    w = X[:, 2]
    psi = X[:, 6]
    r = X[:, 5]
    phi = X[:, 8]
    p = X[:, 3]
    theta = X[:, 7]
    q = X[:, 4]
    name = 'Clasificados_'+str(C)+''+str(n)+''+str(normal_)
    np.savez(name, z, w, psi, r, phi, p, theta, q, cluster)
    postion_vs_velocity(z, w, psi, r, phi, p, theta, q, cluster)
    bar1.finish()
    if not normal_:
        a, b = [i for i, e in enumerate(perturbations) if e != 0]
        save = 'perturbaciones_uniforme_'+str(n)+'_'+str(a)+str(b)+'.png'
        plt.savefig(save)


if __name__ == "__main__":
    dist = np.array([0, 0, 8, .15, .15, 1, 1.5, .2, .2, 0, 0, 20])
    Ze = (10, 0, 0, 0)   
    # du, dv, dw, dp, dq, dr, dpsi, dtheta, dphi, dx, dy, dz
    # perturbations = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1] # todas 
    # perturbations = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  # pitch
    # perturbations = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1] # z
    # perturbations = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # roll
    # perturbations = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] # yaw

    perturbations = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  # roll
    n_tests(Ze, 5000, perturbations, normal_=False, d=dist) # uniforme

    perturbations = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # pitch
    n_tests(Ze, 5000, perturbations, normal_=False, d=dist) # uniforme

