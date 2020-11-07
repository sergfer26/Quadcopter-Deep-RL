#!/usr/bin/env python3
import numpy as np
from Linear.ecuaciones_drone import jac_f
from Linear.step import step, control_feedback, simulador
from DDPG.env.quadcopter_env import D, funcion
from numpy import pi
from numpy.random import uniform as unif
import csv


class Struct_p():
    def __init__(self):
        self.u, self.v, self.w = 0, 0, 0
        self.x, self.y, self.z = 0, 0, 0
        self.p, self.q, self.r = 0, 0, 0
        self.psi, self.theta, self.phi = 0, 0, 0

    def output(self):
        tem = np.array([self.u, self.v, self.w ,self.x, self.y, self.z, self.p, self.q, self.r, self.psi, self.theta, self.phi])
        self.__init__()
        return tem

    
goal = np.array([0, 0, 0, 15, 15, 15, 0, 0, 0, 0, 0, 0])

Ze = (15, 0, 0, 0)
un_grado = np.pi/180.0
p = Struct_p()
p.u = 0.5; p.v = 0.5; p.w = 0.5
p.x = 1; p.y = 1; p.z = 1
p.p = 0.1 * un_grado; p.q = 0.1 * un_grado; p.r = 0.1 * un_grado
p.theta = 5 * un_grado; p.psi = 5 * un_grado; p.phi = 5 * un_grado
perturbacion = p.output()


def vuelos(numero):
    muestra =  []
    for _ in range(numero):
        Y = np.array([g + unif(-e, e) for e, g in zip(perturbacion, goal)])
        vuelo, acciones = simulador(Y, Ze, 30, 800, jac=jac_f)
        for estado, accion in zip(vuelo, acciones):
            # accion = np.abs(accion) porque le quito W0
            nuevo_estado = funcion(estado)
            muestra.append(np.concatenate((accion, nuevo_estado), axis=None))
    with open('tabla_1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(muestra)

vuelos(3)
