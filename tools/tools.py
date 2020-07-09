import numpy as np
from matplotlib import pyplot as plt

def imagen2d(z, w, psi, r, phi, p, theta, q, t):
    f, ((w1, w2), (r1, r2), (p1, p2), (q1, q2)) = plt.subplots(4, 2)
    cero = np.zeros(len(z))

    w1.plot(t, z, c='b',label = str(round(z[-1],4)))
    w1.set_ylabel('z')
    w2.plot(t, w, c='b',label = str(round(w[-1],4)))
    w2.set_ylabel('dz')
    w1.legend()
    w2.legend()

    r1.plot(t, psi, c='r')
    r1.set_ylabel('$\psi$')
    r2.plot(t, r, c='r')
    r2.set_ylabel('d$\psi$')

    p1.plot(t, phi, c='g')
    p1.set_ylabel('$\phi$')
    p2.plot(t, p, c='g')
    p2.set_ylabel(' d$\phi$')

    q1.plot(t, theta)
    q1.set_ylabel('$ \\theta$')
    q2.plot(t, q)
    q2.set_ylabel(' d$ \\theta$')

    w1.plot(t, cero + 15, '--', c='k', alpha=0.5)
    w2.plot(t, cero, '--', c='k', alpha=0.5)
    r2.plot(t, cero, '--', c='k', alpha=0.5)
    p2.plot(t, cero, '--', c='k', alpha=0.5)
    q2.plot(t, cero, '--', c='k', alpha=0.5)

    plt.show()
