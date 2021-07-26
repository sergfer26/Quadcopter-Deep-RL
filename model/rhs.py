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
