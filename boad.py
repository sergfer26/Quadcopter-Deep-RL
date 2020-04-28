#!/usr/bin/env python3
from progress.bar import Bar, ChargingBar
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from numpy import abs
from ecuaciones_drone import f, escribe, imagen, imagen2d, I, K, B, M, L, G
from numpy import pi
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from numpy.linalg import norm
omega_0 = np.sqrt((G * M)/(4 * K))
Ixx, Iyy, Izz = I
F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T  # matriz de control z
F2 = np.array([[0.5, 0, 0.5, 0], [1, 0, 1, 0]]).T  # matriz de control yaw
F3 = np.array([[0, 1, 0, 0.75], [0, 0.5, 0, -0.5]]).T  # matriz de control roll
F4 = np.array([[1, 0, 0.75, 0], [0.5, 0, -0.5, 0]]).T  # matriz de control pitch

c1 = 1  # -(((2*K)/M) * omega_0)**(-1)
c3 = (((L * B) / Ixx) * omega_0)**(-1)
c4 = (((L * B) / Iyy) * omega_0)**(-1)
c2 = (((2 * B) / Izz) * omega_0)**(-1)




def step(W, y_t, t, z_e):
    '''
    Obtiene una solución numérica de dy=f(y) para un tiempo t+1

    param W: arreglo de velocidades de los  4 rotores
    param y: arreglo de 12 posiciones del cuadricoptero
    param t: un intervalo de tiempo

    regresa: y para el siguiente paso de tiempo,el reward
    '''
    #import pdb; pdb.set_trace()
    w1, w2, w3, w4 = W
    delta_y = odeint(f, y_t, t, args=(w1, w2, w3, w4))[1]
    _, _, w, p, q, r, psi, theta, phi, _, _, z = delta_y
    reward = -np.linalg.norm([w, p, q, r, psi, theta, phi,z-z_e])
    return delta_y,reward


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

def simulador(Y, z_e, T, tam):
    '''
    Soluciona el sistema de EDO usando controles en el
    intervalo [0, T].

    param Y: arreglo de la condición inicial del sistema
    param Ze: arreglo de las posiciones estables de los 4 controles
    param T: tiempo final
    param tam: número de elementos de la partición de [0, T]

    regresa; arreglo de la posición final
    '''
    writer = SummaryWriter()
    W0 = np.array([1, 1, 1, 1]).reshape((4, 1)) * omega_0
    X = np.zeros((tam, 12))
    X[0] = Y
    z_inicial = Y[-1]
    t = np.linspace(0, T, tam)
    R = [0]
    done = False
    for i in range(len(t)-1):
        _, _, w, p, q, r, psi, theta, phi, _, _, z = Y
        W1 = control_feedback(z - z_e, w, F1) * c1  # control z
        W2 = control_feedback(psi, r, F2) * c2  # control yaw
        W3 = control_feedback(phi, p, F3) * c3  # control roll
        W4 = control_feedback(theta, q, F4) * c4  # control pitch
        W = W0 + W1 + W2 + W3 + W4
        Y,reward = step(W, Y, [t[i], t[i+1]],z_e)
        R.append(reward)
        X[i+1] = Y
        writer.add_scalar('t vs z', z, t[i])
        writer.add_scalars('Rotores', {'W_0':float(W[0]),'W_1':float(W[1]),'W_2': float(W[2]),'W_3':float(W[3])}, t[i])
        if z > 2*z_inicial:
            print('El dron se estrello contra el suelo',z,reward,done)
            done = True
            print('debe salir')
            return X[0:i],R[0:i]
    writer.close()
    return X,R

if __name__ == "__main__":
    '''
Vamos a suponer que el dron siempre empeiza en 10 mas una perturbacion aleatoria,
y que queremos estabilizar en ze
    '''
    T = 60 #120
    tam = 1500
    ale = np.random.uniform(-5,5)
    z_0 = 10
    Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, pi/30, 0, 0, z_0])
    z_e = 5
    X,R = simulador(Y, z_e, T, tam)
    t = np.linspace(0, T, tam)[0:len(X)]
    z = X[:, 11]
    y = X[:, 10]
    x = X[:, 9]
    phi = X[:, 8]
    theta = X[:, 7]
    psi = X[:, 6]
    r = X[:, 5]
    q = X[:, 4]
    p = X[:, 3]
    w = X[:, 2]
    plt.plot(t,z)
    #plt.show()
    # escribe(x, y, z, psi, theta, phi) #Escribe para que blender lea
    # imagen(x, y, z)
    # input()
    # imagen2d(z, w, psi, r, phi, p, theta, q, t)
