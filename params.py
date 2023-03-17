
PARAMS_ENV = {'TIME_MAX': 10, 'STEPS': 251, 'omega0_per': 0.60,
              'K1': '100', 'K11': '100', 'K2': '1000', 'K21': '100', 'K3': '5'}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': True}

PARAMS_TRAIN_GPS = {'UPDATES': 10, 'N': 8, 'M': 100,
                    'SHOW': False, 'rollouts': 100, 'samples': 3}

PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '3', '$y$': '3', '$z$': '3',
              '$p$': '0.0', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/64', r'$\theta$': 'np.pi/64',
              '$\phi$': 'np.pi/64'}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
