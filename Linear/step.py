#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from time import process_time
from .equations import f, jac_f, escribe, imagen, imagen2d
from numpy import pi
from .constants import CONSTANTS, F, C

G = CONSTANTS['G']
Ixx = CONSTANTS['Ixx']
Iyy = CONSTANTS['Iyy']
Izz = CONSTANTS['Izz']
B = CONSTANTS['B']
M = CONSTANTS['M']
L = CONSTANTS['L']
K = CONSTANTS['K']


omega_0 = np.sqrt((G * M)/(4 * K))
F1, F2, F3, F4 = F
c1, c2, c3, c4 = C
W0 = np.array([1, 1, 1, 1]).reshape((4,)) * omega_0


def imagen_accion(A, t, show=True, dir_=None):
    A = np.array(A)
    t = t[0:len(A)]
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, A[:, 0], c='b')
    ax[0].set_ylabel('$a_1$')
    ax[1].plot(t, A[:, 1], c='r')
    ax[1].set_ylabel('$a_2$')
    ax[2].plot(t, A[:, 2], c='g')
    ax[2].set_ylabel('$a_3$')
    ax[3].plot(t, A[:, 3])
    ax[3].set_ylabel('$a_4$')
    if show:
        plt.show()
    else:
        plt.savefig(dir_+'/actions.png')


def step(W, y, t, jac=None):
    '''
    Obtiene una solución numérica de dy=f(y) para un tiempo t+1

    param W: arreglo de velocidades de los  4 rotores
    param y: arreglo de 12 posiciones del cuadricoptero
    param t: un intervalo de tiempo [t0, t1]

    regresa: y para el siguiente paso de tiempo
    '''
    w1, w2, w3, w4 = W
    return odeint(f, y, t, args=(w1, w2, w3, w4), Dfun=jac)


def control_feedback(x, y, F):
    '''
    Realiza el control lineal de las velocidades W
    dadas las variables (x, y).

    param x: variable independiente (dx = y)
    param y: variable dependiente
    param F: matriz 2x4 de control

    regresa: W = w1, w2, w3, w4
    '''
    A = np.array([x, y]).reshape((2, 1))
    return np.dot(F, A)


def simulador(Y, Ze, T, tam, jac=None):
    '''
    Soluciona el sistema de EDO usando controles en el
    intervalo [0, T].

    param Y: arreglo de la condición inicial del sistema
    param Ze: arreglo de las posiciones estables de los 4 controles
    param T: tiempo final
    param tam: número de elementos de la partición de [0, T]

    regresa; arreglo de la posición final
    '''
    z_e, psi_e, phi_e, theta_e = Ze
    X = np.zeros((tam, 12))
    X[0] = Y
    t = np.linspace(0, T, tam)
    acciones = []
    for i in range(len(t)-1):
        u, v, w, x, y, z, p, q, r, psi, theta, phi = Y
        W1 = control_feedback(z - z_e, w, F1) * (c1 ** 2)  # control z
        W2 = control_feedback(psi - psi_e, r, F2) * c2  # control yaw
        W3 = control_feedback(phi - phi_e, p, F3) * c3  # control roll
        W4 = control_feedback(theta - theta_e, q, F4) * c4  # control pitch
        W = W0 + W1 + W2 + W3 + W4
        tem = W1 + W2 + W3 + W4
        acciones.append(tem)
        Y = step(W, Y, [t[i], t[i+1]], jac=jac)[1]

        X[i+1] = Y
    return X, acciones


def nsim3D(n):
    degree = pi/180
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    high = np.array([0, 0, 1, 0, 0, 10, 0.01 * degree, 0.01 * degree,
                    0.01 * degree, 10 * degree, 10 * degree, 10 * degree])
    low = np.array([-0, -0, -1,  -0, -0, -10, -0.01 * degree, -0.01 *
                   degree, -0.01 * degree, -10 * degree, -10 * degree, -10 * degree])
    t = np.linspace(0, 30, 800)
    for _ in range(n):
        state = np.array([np.random.uniform(x, y) for x, y in zip(low, high)])
        X, Y, Z = [], [], []
        for i in range(len(t)-1):
            u, v, w, x, y, z, p, q, r, psi, theta, phi = state
            W1 = control_feedback(z, w, F1) * c1  # control z
            W2 = control_feedback(psi, r, F2) * c2  # control yaw
            W3 = control_feedback(phi, p, F3) * c3  # control roll
            W4 = control_feedback(theta, q, F4) * c4  # control pitch
            W = W0 + W1 + W2 + W3 + W4
            Z.append(z)
            X.append(x)
            Y.append(y)
            new_state = step(W, state, [t[i], t[i+1]])[1]
            state = new_state
        ax.plot(X, Y, Z, '.b', alpha=0.3, markersize=1)
    fig.suptitle('Vuelos', fontsize=16)
    ax.plot(0, 0, 0, '.r', alpha=0.3, markersize=1)
    plt.show()


'''
T = 10
tam = 2600
un_grado = np.pi/180.0
high = np.array([0.0, 0.0, 0.0, 0, 0, 0.5, 0.00, 0.00, 0.00, 0, 0, 0])
low = np.array([0.0, 0.0, 0.0,  0, 0, -0.5, 0, -
               0.0, 0, 0, 0, 0])
Y = np.array([np.random.uniform(x, y) for x, y in zip(low, high)])
Ze = (0, 0, 0, 0)
start = process_time()
X, A = simulador(Y, Ze, T, tam, jac=jac_f)
t = np.linspace(0, T, tam)
w = X[:, 2]
x = X[:, 3]
y = X[:, 4]
z = X[:, 5]
p = X[:, 6]
q = X[:, 7]
r = X[:, 8]
psi = X[:, 9]
theta = X[:, 10]
phi = X[:, 11]
# escribe(x, y, z, psi, theta, phi) #Escribe para que blender lea
# imagen(x, y, z)
imagen2d(z, w, psi, r, phi, p, theta, q, t)
imagen_accion(A, t)
'''
# nsim3D(10)'''
