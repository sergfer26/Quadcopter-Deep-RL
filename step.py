
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from numpy import sin
from numpy import cos
from numpy import tan
from ecuaciones_drone import *

G = 9.81
I = (4.856*10**-3, 4.856*10**-3, 8.801*10**-3)
B, M, L = 1.140*10**(-6), 1.433, 0.225
K = 0.001219 #kt

w1,w2,w3,w4 = (0, 53.6666, 55.6666, 1)
y = 0, 0, 0, 0, 0, 0, 2, 3, 10, 0, 0, 10
W = [w1,w2,w3,w4]

def step(W, y, t):
    w1,w2,w3,w4 = W
    return odeint(f, y, t, args = (w1,w2,w3,w4))

def Simulador(y,T,tam): #Esta funcion permite solucionar la EDO con controles
    X = np.zeros((tam,12))
    X[0] = y
    t = np.linspace(0, T, tam)
    for i in range(len(t)-1):
        y = step(W,y,[t[i],t[i+1]])[1]
        X[i+1] = y
    return X

X = Simulador(y,30,3000)#Contiene las 12 variables

z = X[:,11]
y = X[:,10]
x = X[:,9]
psi = X[:,6]
theta = X[:,7]
phi = X[:,8]

escribe(x,y,z,psi,theta,phi)#Escribe para que blender pueda leer
