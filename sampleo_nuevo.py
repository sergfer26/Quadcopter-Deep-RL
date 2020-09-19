#!/usr/bin/env python3
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import tan
from ecuaciones_drone import f,jac_f, escribe, imagen, imagen2d, I, K, B, M, L, G
from numpy import pi
from numpy.linalg import norm
from numpy.random import uniform as unif
import csv
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



class Struct_p():
    def __init__(self):
        self.u, self.v, self.w = 0,0,0
        self.x, self.y, self.z = 0,0,0
        self.p, self.q, self.r = 0,0,0
        self.psi, self.theta, self.phi = 0,0,0


    def output(self):
        tem = np.array([self.u, self.v, self.w ,self.x, self.y, self.z, self.p, self.q, self.r, self.psi, self.theta, self.phi])
        self.__init__()
        return tem
    
goal = np.array([0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0])

def step(W, y, t,jac=None):
    '''
    Obtiene una solución numérica de dy=f(y) para un tiempo t+1

    param W: arreglo de velocidades de los  4 rotores
    param y: arreglo de 12 posiciones del cuadricoptero
    param t: un intervalo de tiempo

    regresa: y para el siguiente paso de tiempo
    '''
    #import pdb; pdb.set_trace()
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
    rewards =  [] 
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
    return X,acciones


def D(angulos):
    z, y, x = angulos # psi, theta, phi
    R = np.array([
        [cos(z) * cos(y), cos(z) * sin(y) * sin(x) - sin(z) * cos(x), cos(z) * sin(y) * cos(x) + sin(z) * sin(x)],
        [sin(z) * cos(y), sin(z) * cos(y) * sin(x) + cos(z) * cos(x), sin(z) * sin(y) * cos(x) - cos(z) * sin(x)], 
        [- sin(y), cos(y) * sin(x), cos(y) * cos(x)]
    ])
    return R


def funcion(state):
    angulos = state[9:]
    state = state[0:9]
    orientacion = np.matrix.flatten(D(angulos))
    return np.concatenate([state, orientacion])




Ze = (15, 0, 0, 0)
un_grado = np.pi/180.0
p = Struct_p()
p.u = 1; p.v = 1; p.w = 1;
p.x = 10; p.y = 10; p.z = 7;
p.p = 0.01*un_grado; p.q = 0.01*un_grado; p.r =  0.01*un_grado
p.theta = 10*un_grado; p.psi = 10*un_grado; p.phi = 10*un_grado
perturbacion = p.output()


def vuelos(numero):
    muestra =  []
    for _ in range(numero):
        Y = np.array([g + unif(-e, e) for e, g in zip(perturbacion, goal)])
        vuelo,acciones = simulador(Y, Ze, 30, 800,jac=jac_f)
        for estado,accion in zip(vuelo,acciones):
            nuevo_estado = funcion(estado)
            muestra.append( np.concatenate((accion,nuevo_estado), axis=None))
    matriz = np.reshape(muestra,(799*numero,22))
    with open('tablaaaaaaa.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(muestra)

vuelos(3)
