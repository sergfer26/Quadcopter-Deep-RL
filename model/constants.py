import numpy as np
from sympy import symbols
from struct_var import Struct


m, s, radians = symbols('m s radians')


CONSTANTS = {
    'G': Struct(typ='Cnts', varid='G', prn='$G$', desc='Gravity constant',
                units=m * s**-2, val=9.81, rec=1),
    'Ix': Struct(typ='Cnts', varid='Ix', prn='$I_x$',
                 desc='Inertia tensor component x', units=1, val=4.856*10**-3,
                 rec=1),
    'Iy': Struct(typ='Cnts', varid='Iy', prn='$I_y$',
                 desc='Inertia tensor component y', units=1, val=4.856*10**-3,
                 rec=1),
    'Iz': Struct(typ='Cnts', varid='Iz', prn='$I_z$',
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
                      val=np.sqrt((9.81 * 1.433)/(4 * 0.001219)), rec=1)  # G, M, _, K
}


STATE_VARS = {
    'x': Struct(typ='State', varid='x', prn='$x$', desc='Position x of the Quadcopter',
                units=m, val=0.0, rec=1),
    'y': Struct(typ='State', varid='y', prn='$y$', desc='Position y of the Quadcopter',
                units=m, val=0.0, rec=1),
    'z': Struct(typ='State', varid='z', prn='$z$', desc='Position z of the Quadcopter',
                units=m, val=0.0, rec=1),
    'dx': Struct(typ='State', varid='dx', prn='$dx$', desc='Derivative of x',
                 units=m, val=0.0, rec=1),
    'dy': Struct(typ='State', varid='dy', prn='$dy$', desc='Derivative of y',
                 units=m, val=0.0, rec=1),
    'dz': Struct(typ='State', varid='dz', prn='$dz$', desc='Derivative of x',
                 units=m, val=0.0, rec=1),
    'psi': Struct(typ='State', varid='psi', prn='$psi$', desc='Yaw',
                  units=radians, val=0.0, rec=1),
    'theta': Struct(typ='State', varid='theta', prn='$theta$', desc='Pitch',
                    units=radians, val=0.0, rec=1),
    'phi': Struct(typ='State', varid='phi', prn='$phi$', desc='Roll',
                  units=radians, val=0.0, rec=1),
    'dpsi': Struct(typ='State', varid='dpsi', prn='$dpsi$', desc='Derivative of Yaw',
                   units=radians, val=0.0, rec=1),
    'dtheta': Struct(typ='State', varid='dtheta', prn='$dtheta$', desc='Derivative of Pitch',
                     units=radians, val=0.0, rec=1),
    'dphi': Struct(typ='State', varid='dphi', prn='$dphi$', desc='Derivative of Roll',
                   units=radians, val=0.0, rec=1)
}
