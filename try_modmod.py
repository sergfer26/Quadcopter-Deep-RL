from mod_mod.model import Quadcopter
from Linear.agent import LinearAgent
from env import QuadcopterEnv
from params import STATE_NAMES

env = QuadcopterEnv()
agent = LinearAgent(env)
agent.vars = ['u', 'v', 'w', 'x', 'y', 'z',
              'p', 'q', 'r', 'phi', 'theta', 'psi']
director = Quadcopter(agent)
director.Run(Dt=0.04, n=375, sch=['Dynamics'], active=True)
breakpoint()
# director.V_GetRec('u')
