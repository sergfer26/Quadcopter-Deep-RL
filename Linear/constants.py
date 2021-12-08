
import numpy as np

'''
CONSTANTS = {'G': 9.81, 'Ixx': 1, 'Iyy': 1, 'Izz': 0.5,
             'B': 1.140 * 1e-7, 'M': 1, 'L': 0.225, 'K': 2.980 * 1e-6}
'''

CONSTANTS = {'G': 9.81, 'Ixx': 4.856 * 1e-3, 'Iyy': 4.856 * 1e-3,
             'Izz': 8.801 * 1e-3, 'B': 1.140 * 1e-6, 'M': 1.433,
             'L': 0.225, 'K': 0.001219}

G, Ixx, Iyy, Izz, B, M, L, K = CONSTANTS.values()

omega_0 = np.sqrt((G * M)/(4 * K))

W0 = np.array([1, 1, 1, 1]).reshape((4, 1)) * omega_0

F1 = np.array([[0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]]).T  # control z
F2 = np.array([[0.5, 0, 0.5, 0], [1, 0, 1, 0]]).T  # control yaw
F3 = np.array([[0, 1, 0, 0.75], [0, 0.5, 0, -0.5]]).T  # control roll
F4 = np.array([[1, 0, 0.75, 0], [0.5, 0, -0.5, 0]]).T  # control pitch


c1 = (((2*K)/M) * omega_0)**(-1)  # 1z
c3 = (((L * B) / Ixx) * omega_0)**(-1)  # roll
c4 = (((L * B) / Iyy) * omega_0)**(-1)  # pitch
c2 = (((2 * B) / Izz) * omega_0)**(-1)  # yaw

C = np.array([c1, c2, c3, c4])
F = np.array([F1, F2, F3, F4])
