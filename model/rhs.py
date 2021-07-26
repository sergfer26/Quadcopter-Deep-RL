from ModMod import StateRHS
from .constants import CONSTANTS
from sympy import symbols
from numpy import sin, cos, tan


mt = symbols('mt')


class psi_rhs(StateRHS):
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


class theta_rhs(StateRHS):
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


class phi_rhs(StateRHS):
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


class x_rhs(StateRHS):
    """Define a RHS, this is the rhs for x"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        CONSTANTS['u'].addvar_rhs(self)

    def RHS(self, Dt):
        u_ = self.V('u')
        return u_


class y_rhs(StateRHS):
    """Define a RHS, this is the rhs for y"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        CONSTANTS['v'].addvar_rhs(self)

    def RHS(self, Dt):
        v_ = self.V('v')
        return v_


class z_rhs(StateRHS):
    """Define a RHS, this is the rhs for z"""

    def __init__(self):
        """Define a RHS, ***this an assigment RHS***, V1 = h2(...), NO ODE."""
        super().__init__()
        self.SetSymbTimeUnits(mt)  # minuts
        CONSTANTS['w'].addvar_rhs(self)

    def RHS(self, Dt):
        w_ = self.V('w')
        return w_
