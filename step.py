#!/usr/bin/env python3
import numpy as np
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from ecuaciones_drone import f, escribe, imagen, imagen2d 


F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T # matriz de control z
F2 = np.array([[0.5, 0, 0.5, 0], [1, 0, 1, 0]]).T # matriz de control yaw
F3 = np.array([[0, 1, 0, 0.75], [0, 0.5, 0, -0.5]]).T # matriz de control roll
F4 = np.array([[1, 0, 0.75, 0], [0.5, 0, -0.5, 0]]).T # matriz de control pitch


def step(W, y, t):
    '''
    Obtiene una solución numérica de dy=f(y) para un tiempo t+1

    param W: arreglo de velocidades de los  4 rotores
    param y: arreglo de 12 posiciones del cuadricoptero
    param t: un paso de tiempo

    regresa: y para el tiempo t+1 
    '''
    w1, w2, w3, w4 = W
    return odeint(f, y, t, args=(w1, w2, w3, w4))


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


def simulador(y, T, tam):
    '''
    Soluciona el sistema de EDO usando controles en el
    intervalo [0, T].

    param y: arreglo de la condición inicial del sistema
    param T: tiempo final
    param tam: número de elementos de la partición de [0, T]

    regresa; arreglo de la posición final
    '''
    W0 = np.array([1, 1, 1, 1])
    X = np.zeros((tam, 12))
    X[0] = y
    t = np.linspace(0, T, tam)
    for i in range(len(t)-1):
        _, _, w, p, q, r, psi, theta, phi, _, _, z = y
        W1 = control_feedback(z, w, F1) # control z
        W2 = control_feedback(psi, r, F2)  # control yaw
        W3 = control_feedback(phi, p, F3) # control roll
        W4 = control_feedback(theta, q, F4) # control pitch
        W = W0 + W1 + W2 + W3 + W4
        y = step(W, y, [t[i], t[i+1]])[1]
        X[i+1] = y
    return X

if __name__ == "__main__":
    T = 7
    tam = 4000
    X = simulador(y, T, tam)
    t = np.linspace(0, T, tam)
    z = X[:, 11]
    y = X[:, 10]
    x = X[:, 9]
    phi = X[:, 8]
    theta = X[:, 7]
    psi = X[:, 6]
    r = X[:, 5]
    q = X[:, 4]
    p = X[:, 3]
    w = X[:,2]
    #escribe(x, y, z, psi, theta, phi)#Escribe para que blender lea
    #imagen(x, y, z)
    #input()
    imagen2d(z, w, psi, r, phi, p, theta, q, t)
