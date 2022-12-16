import numpy as np
from ModMod import Module, Director
from .rhs import U_rhs, V_rhs, W_rhs, X_rhs, Y_rhs, Z_rhs
from .rhs import P_rhs, Q_rhs, R_rhs, Psi_rhs, Phi_rhs, Theta_rhs


rhs_ins = {'u': U_rhs(),
           'v': V_rhs(),
           'w': W_rhs(),
           'x': X_rhs(),
           'y': Y_rhs(),
           'z': Z_rhs(),
           'p': P_rhs(),
           'q': Q_rhs(),
           'r': R_rhs(),
           'psi': Psi_rhs(),
           'phi': Phi_rhs(),
           'theta': Theta_rhs()
           }


class Dynamics(Module):
    def __init__(self, Dt=0.1):
        super().__init__(Dt)  # Time steping of module
        for k, v in rhs_ins.items():
            self.AddStateRHS(k, v)

    def Advance(self, t1):
        self.AdvanceRungeKutta(t1, Method=4, t0=0.0)
        self.AdvanceAssigment(t1)
        return 1


class Quadcopter(Director):
    def __init__(self, agent, t0=0.0, time_unit="s", Vars={},
                 Modules=dict(),
                 units_symb=True):
        super().__init__(t0=t0, time_unit=time_unit, Vars=Vars,
                         Modules=Modules)
        self.MergeVarsFromRHSs(list(rhs_ins.values()), call=__name__)
        self.AddModule('Dynamics', Dynamics())
        self.agent = agent

    def _get_vars(self):
        '''
        Sirve para obtener todas las variables y las guarda en un diccionario
        '''
        return {id: Obj.val for id, Obj in self.Vars.items()}

    def _get_state(self):
        Vars = self._get_vars()
        partial_vars = {id: Vars[id] for id in self.agent.vars}
        state = np.array(list(partial_vars.values()))
        return state

    def _update_controls(self, **controls):
        '''
        Actualiza los controles.
        controls = {'U1': u1, etc}
        '''
        for k, v in controls.items():
            self.V_Set(k, v)  # Set en variables de director

    def Scheduler(self, t1, sch):
        state = self._get_state()
        w1, w2, w3, w4 = self.agent.get_action(state)
        self._update_controls(w1=w1, w2=w2, w3=w3, w4=w4)
        return super().Scheduler(t1, sch)
