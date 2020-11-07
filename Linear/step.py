#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from time import process_time 
from .ecuaciones_drone import f, jac_f, escribe, imagen, imagen2d, I, K, B, M, L, G
from numpy import pi

omega_0 = np.sqrt((G * M)/(4 * K))
Ixx, Iyy, Izz = I
F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T  # matriz de control z
F2 = np.array([[0.5, 0, 0.5, 0], [1, 0, 1, 0]]).T  # matriz de control yaw
F3 = np.array([[0, 1, 0, 0.75], [0, 0.5, 0, -0.5]]).T  # matriz de control roll
F4 = np.array([[1, 0, 0.75, 0], [0.5, 0, -0.5, 0]]).T  # matriz de control pitch

c1 = (((2*K)/M) * omega_0)**(-1) 
c3 = (((L * B) / Ixx) * omega_0)**(-1)
c4 = (((L * B) / Iyy) * omega_0)**(-1)
c2 = (((2 * B) / Izz) * omega_0)**(-1)


def imagen_accion(A, t, show=True, dir_=None):
    A = np.array(A)
    t = t[0:len(A)]
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, A[:,0], c='b')
    ax[0].set_ylabel('$a_1$')
    ax[1].plot(t, A[:,1], c='r')
    ax[1].set_ylabel('$a_2$')
    ax[2].plot(t, A[:,2], c='g')
    ax[2].set_ylabel('$a_3$')
    ax[3].plot(t, A[:,3])
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
    param t: un intervalo de tiempo

    regresa: y para el siguiente paso de tiempo
    '''
    w1, w2, w3, w4 = W
    return odeint(f, y, t, args=(w1, w2, w3, w4) ,Dfun=jac)


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


def simulador(Y, Ze, T, tam,jac=None):
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
    W0 = np.array([1, 1, 1, 1]).reshape((4, 1)) * omega_0
    X = np.zeros((tam, 12))
    X[0] = Y
    t = np.linspace(0, T, tam)
    acciones = []
    for i in range(len(t)-1):
        u, v, w, x, y, z, p, q, r, psi, theta, phi = Y
        W1 = control_feedback(z - z_e, w, F1) * c1  # control z
        W2 = control_feedback(psi - psi_e, r, F2) * c2  # control yaw
        W3 = control_feedback(phi - phi_e, p, F3) * c3  # control roll
        W4 = control_feedback(theta - theta_e, q, F4) * c4  # control pitch
        W = W0 + W1 + W2 + W3 + W4
        tem = W1 + W2 + W3 + W4
        acciones.append(tem)
        Y = step(W, Y, [t[i], t[i+1]],jac=jac)[1]

        X[i+1] = Y
    return X, acciones 

'''
T = 120
tam = 3200
un_grado = np.pi/180.0
Y = np.zeros(12)
Y[5] = 10 
Ze = (15, 0, 0, 0)
start = process_time() 
X, A = simulador(Y, Ze, T, tam,jac=jac_f)
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
#escribe(x, y, z, psi, theta, phi) #Escribe para que blender lea
#imagen(x, y, z)
imagen2d(z, w, psi, r, phi, p, theta, q, t)
#imagen_accion(A,t)
'''
    
    
