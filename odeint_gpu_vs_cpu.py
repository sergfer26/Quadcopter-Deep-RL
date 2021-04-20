import numpy as np
import numba
import timeit
from Linear.equations import f, jac_f, omega_0
from scipy.integrate import odeint

time = np.linspace(0, 30, 800)
t = [time[0], time[1]]
numba_f = numba.jit(f)
numba_jac = numba.jit(jac_f)
x = np.zeros(12)
W0 = (omega_0, omega_0, omega_0, omega_0)


def time_func1():
    sol = odeint(f, x, t, args=W0)


def time_func2():
    sol = odeint(numba_f, x, t, args=W0)


def time_func3():
    sol = odeint(f, x, t, args=W0, Dfun=jac_f)


def time_func4():
    sol = odeint(numba_f, x, t, args=W0, Dfun=numba_jac)


t1 = timeit.Timer(time_func1).timeit(number=1000)
t2 = timeit.Timer(time_func2).timeit(number=1000)
t3 = timeit.Timer(time_func3).timeit(number=1000)
t4 = timeit.Timer(time_func4).timeit(number=1000)


print('without gpu time {}'.format(t1))
print('numba time: {}'.format(t2))
print('without gpu/ with jac time {}'.format(t3))
print('numba with jac time: {}'.format(t4))
