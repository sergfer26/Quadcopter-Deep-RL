from ecuaciones_drone import f
from scipy.integrate import odeint


def step(W, X, delta_t):
    return odeint(f, X, delta_t, args=(W))
