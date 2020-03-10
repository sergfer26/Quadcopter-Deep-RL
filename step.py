from ecuaciones_drone import f
from scipy.integrate import odeint


def step(W, X, delta_t):
    sol = odeint(f, X, delta_t, args=(W))
    psi = sol[:,6]
    theta = sol[:,7]
    phi = sol[:,8]
    x = sol[:,9]
    y = sol[:,10]
    z = sol[:,11]
    return psi, theta, phi, x, y, z
