import numpy as np
from ModMod import StateRHS
from .constants import CONSTANTS
from sympy import symbols
from numpy import sin, cos, tan
from numpy.linalg import norm


mt = symbols('mt')


class X_rhs(StateRHS):
    """Define a RHS, this is the rhs for x"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        CONSTANTS['u'].addvar_rhs(self)

    def RHS(self, Dt):
        u_ = self.V('u')
        return u_


class Y_rhs(StateRHS):
    """Define a RHS, this is the rhs for y"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        CONSTANTS['v'].addvar_rhs(self)

    def RHS(self, Dt):
        v_ = self.V('v')
        return v_


class Z_rhs(StateRHS):
    """Define a RHS, this is the rhs for z"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        CONSTANTS['w'].addvar_rhs(self)

    def RHS(self, Dt):
        w_ = self.V('w')
        return w_


class U_rhs(StateRHS):
    """Define a RHS, this is the rhs for u"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['r', 'v', 'q', 'w', 'G', 'theta']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        r_ = self.V('r')
        v_ = self.V('v')
        q_ = self.V('q')
        w_ = self.V('w')
        G_ = self.V('G')
        theta_ = self.V('theta')
        u_ = r_ * v_ - q_ * w_ - G_ * sin(theta_)
        return u_


class V_rhs(StateRHS):
    """Define a RHS, this is the rhs for v"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['p', 'w', 'r', 'u', 'G', 'theta', 'phi']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        p_ = self.V('p')
        w_ = self.V('w')
        r_ = self.V('r')
        u_ = self.V('u')
        G_ = self.V('G')
        theta_ = self.V('theta')
        phi_ = self.V('phi')
        v_ = p_ * w_ - r_ * u_ - G_ * cos(theta_) * sin(phi_)
        return v_


class W_rhs(StateRHS):
    """Define a RHS, this is the rhs for w"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['q', 'u', 'p', 'v', 'G', 'phi', 'theta', 'K', 'M', 'w1', 'w2',
                'w3', 'w4']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        p_ = self.V('p')
        u_ = self.V('u')
        q_ = self.V('q')
        v_ = self.V('v')
        G_ = self.V('G')
        theta_ = self.V('theta')
        phi_ = self.V('phi')
        K_ = self.V('K')
        M_ = self.V('M')
        w1_ = self.V('w1')
        w2_ = self.V('w2')
        w3_ = self.V('w3')
        w4_ = self.V('w4')
        W_ = np.array([w1_, w2_, w3_, w4_])
        w_ = q_ * u_ - p_ * v_ + G_ * \
            cos(phi_) * cos(theta_) - (K_/M_) * norm(W_) ** 2
        return w_


class Psi_rhs(StateRHS):
    """Define a RHS, this is the rhs for  dpsi"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['q', 'r', 'phi', 'theta']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        q_ = self.V('q')
        r_ = self.V('r')
        phi_ = self.V('phi')
        theta_ = self.V('theta')
        dpsi_ = (q_ * sin(self.phi_) + r_ * cos(phi_)) * (1 / cos(theta_))
        return dpsi_


class Theta_rhs(StateRHS):
    """Define a RHS, this is the rhs for theta"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['q', 'r', 'phi']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        q_ = self.V('q')
        r_ = self.V('r')
        phi_ = self.V('phi')
        theta_ = q_ * cos(phi_) - r_ * sin(phi_)
        return theta_


class Phi_rhs(StateRHS):
    """Define a RHS, this is the rhs for phi"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['p', 'q', 'r', 'phi', 'theta']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        p_ = self.V('p')
        q_ = self.V('q')
        r_ = self.V('r')
        phi_ = self.V('phi')
        theta_ = self.V('theta')
        phi_ = p_ + (q_ * sin(phi_) + r_ * cos(phi_)) * tan(theta_)
        return phi_


class P_rhs(StateRHS):
    """Define a RHS, this is the rhs for p"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['L', 'B', 'Ixx', 'phi', 'theta']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        L_ = self.V('L')
        B_ = self.V('B')
        Ixx_ = self.V('Ixx')
        w4_ = self.V('w4')
        w2_ = self.V('w2')
        q_ = self.V('q')
        r_ = self.V('r')
        Izz_ = self.V('Izz')
        Iyy_ = self.V('Iyy')
        p_ = ((L_ * B_) / Ixx_) * (w4_ ** 2 - w2_ ** 2) - \
            q_ * r_ * ((Izz_ - Iyy_) / Ixx_)
        return p_


class Q_rhs(StateRHS):
    """Define a RHS, this is the rhs for p"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['L', 'B', 'Iyy', 'w3', 'w1', 'p', 'r', 'Ixx', 'Izz']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        L_ = self.V('L')
        B_ = self.V('B')
        Ixx_ = self.V('Ixx')
        w3_ = self.V('w4')
        w1_ = self.V('w2')
        q_ = self.V('q')
        p_ = self.V('p')
        r_ = self.V('r')
        Izz_ = self.V('Izz')
        Iyy_ = self.V('Iyy')
        q_ = ((L_ * B_) / Iyy_) * (w3_ ** 2 - w1_ ** 2) - \
            p_ * r_ * ((Ixx_ - Izz_) / Iyy_)
        return q_


class R_rhs(StateRHS):
    """Define a RHS, this is the rhs for p"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        vars = ['L', 'B', 'Iyy', 'w3', 'w1', 'p', 'r', 'Ixx', 'Izz']
        for name in vars:
            CONSTANTS[name].addvar_rhs(self)

    def RHS(self, Dt):
        B_ = self.V('B')
        w1_ = self.V('w1')
        w2_ = self.V('w2')
        w3_ = self.V('w3')
        w4_ = self.V('w4')
        Izz_ = self.V('Izz')
        r_ = (B_/Izz_) * (w2_ ** 2 + w4_ ** 2 - w1_ ** 2 - w3_ ** 2)
        return r_
