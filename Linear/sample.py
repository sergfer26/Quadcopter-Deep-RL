import matplotlib.pyplot as plt
import numpy as np

X = [2, 36, 14.9, -18.6, 2]
Y = [-6, 4, 6.7, -2.7, -6]


def inside(n):  # Muestrea n puntos en la region 'estable' de z con aceptacion rechazo
    puntosx = []
    puntosy = []
    while len(puntosx) != n:
        x = np.random.uniform(-20, 40)
        y = np.random.uniform(-8, 8)
        adentro = True
        for i in range(len(X)-1):
            uno = [1, X[i], Y[i]]
            dos = [1, X[i+1], Y[i+1]]
            tres = [1, x, y]
            matriz = np.array([uno, dos, tres])
            if np.linalg.det(matriz) < 0:
                adentro = False
        if adentro == True:
            puntosx.append(x)
            puntosy.append(y)
    return puntosx, puntosy


x, y = inside(500)
plt.plot(X, Y, 'or')
plt.plot(x, y, '.b', alpha=0.3)
plt.show()
