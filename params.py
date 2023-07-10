
import numpy as np

PARAMS_ENV = {'dt': 0.04, 'STEPS': 375, 'omega0_per': 0.60,
              'K1': '10', 'K11': '10', 'K2': '100', 'K21': '10', 'K3': '.5'}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': True}

PARAMS_TRAIN_GPS = {'UPDATES': 5, 'N': 3, 'M': 400,
                    'SHOW': True, 'is_stochastic': False, 'samples': 3,
                    'batch_size': 20, 'shuffle_batches': True, 'time_step': 1,
                    'policy_updates': 3}

PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '5', '$y$': '5', '$z$': '5',
              '$p$': '0.00', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/64', r'$\theta$': 'np.pi/64',
              '$\phi$': 'np.pi/64'}

PARAMS_DDPG = {'hidden_sizes': [64, 64], 'actor_learning_rate': '1e-2',
               'critic_learning_rate': 1e-4, 'gamma': 0.98, 'tau': 0.125,
               'max_memory_size': int(1e4)}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']

high = np.array([
    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    [5., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 5., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 9., 0., 0., 14., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., .0, .1, 0., 0., 0., .2],
    [0., 0., 0., 0., 0., 0., 0., 0., .1, 0., .2, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., .1, np.pi/4, 0., 0.]
])

low = -high
state_space = np.stack([low, high])


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
