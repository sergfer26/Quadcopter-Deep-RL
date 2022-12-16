import numpy as np
from sympy import symbols
from .struct_var import Struct


m, s, radians = symbols('m s radians')


CONSTANTS = {
    'G': Struct(typ='Cnts', varid='G', prn='$G$', desc='Gravity constant',
                units=m * s**-2, val=9.81, rec=1),
    'Ixx': Struct(typ='Cnts', varid='Ixx', prn='$I_{xx}$',
                  desc='Inertia tensor component x', units=1, val=4.856*10**-3,
                  rec=1),
    'Iyy': Struct(typ='Cnts', varid='Iyy', prn='$I_{yy}$',
                  desc='Inertia tensor component y', units=1, val=4.856*10**-3,
                  rec=1),
    'Izz': Struct(typ='Cnts', varid='Izz', prn='$I_{zz}$',
                  desc='Inertia tensor component z', units=1, val=8.801*10**-3,
                  rec=1),
    'B': Struct(typ='Cnts', varid='B', prn='$B$', desc='Missing', units=1,
                val=1.140*10**(-6), rec=1),
    'M': Struct(typ='Cnts', varid='M', prn='$M$', desc='Missing', units=1,
                val=1.433, rec=1),
    'L': Struct(typ='Cnts', varid='L', prn='$L$', desc='Missing', units=1,
                val=0.225, rec=1),
    'K': Struct(typ='Cnts', varid='K', prn='$K$', desc='Missing', units=1,
                val=0.001219, rec=1),  # kt
    'omega_0': Struct(typ='Cnts', varid='omega_0', prn='$\omega_0$',
                      desc='Missing', units=1,
                      val=np.sqrt((9.81 * 1.433)/(4 * 0.001219)), rec=1),  # G, M, _, K
    # }


    # STATE_VARS = {
    'x': Struct(typ='State', varid='x', prn='$x$', desc='Position x of the Quadcopter',
                units=m, val=0.0, rec=1),
    'y': Struct(typ='State', varid='y', prn='$y$', desc='Position y of the Quadcopter',
                units=m, val=0.0, rec=1),
    'z': Struct(typ='State', varid='z', prn='$z$', desc='Position z of the Quadcopter',
                units=m, val=0.0, rec=1),
    'u': Struct(typ='State', varid='u', prn='$u$', desc='Derivative of x',
                units=m, val=0.0, rec=1),
    'v': Struct(typ='State', varid='v', prn='$v$', desc='Derivative of y',
                units=m, val=0.0, rec=1),
    'w': Struct(typ='State', varid='w', prn='$w$', desc='Derivative of x',
                units=m, val=0.0, rec=1),
    'psi': Struct(typ='State', varid='psi', prn='$psi$', desc='Yaw',
                  units=radians, val=0.0, rec=1),
    'theta': Struct(typ='State', varid='theta', prn='$theta$', desc='Pitch',
                    units=radians, val=0.0, rec=1),
    'phi': Struct(typ='State', varid='phi', prn='$phi$', desc='Roll',
                  units=radians, val=0.0, rec=1),
    'r': Struct(typ='State', varid='r', prn='$r$', desc='Derivative of Yaw',
                units=radians, val=0.0, rec=1),
    'q': Struct(typ='State', varid='q', prn='$q$', desc='Derivative of Pitch',
                units=radians, val=0.0, rec=1),
    'p': Struct(typ='State', varid='p', prn='$p$', desc='Derivative of Roll',
                units=radians, val=0.0, rec=1),
    # }
    #
    #
    # ACTIONS = {
    'w1': Struct(typ='State', varid='w1', prn='$\omega_1$', desc='Control 1',
                 units=1, val=0.0, rec=1),
    'w2': Struct(typ='State', varid='w2', prn='$\omega_2$', desc='Control 2',
                 units=1, val=0.0, rec=1),
    'w3': Struct(typ='State', varid='w3', prn='$\omega_3$', desc='Control 3',
                 units=1, val=0.0, rec=1),
    'w4': Struct(typ='State', varid='w4', prn='$\omega_4$', desc='Control 4',
                 units=1, val=0.0, rec=1)
}
