from .constants import F, C
from .step import control_feedback


F1, F2, F3, F4 = F
c1, c2, c3, c4 = C


class LinearAgent:

    def __init__(self, env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

    def get_action(self, state):
        _, _, w, _, _, z, p, q, r, psi, theta, phi = state
        # _, _, w, _, _, z, p, q, r, phi, theta, psi
        W1 = control_feedback(z, w, F1) * c1   # control z
        W2 = control_feedback(psi, r, F2) * c2  # control yaw
        W3 = control_feedback(phi, p, F3) * c3  # control roll
        W4 = control_feedback(theta, q, F4) * c4  # control pitch
        W = W1 + W2 + W3 + W4
        return W.reshape(4)
