
PARAMS_ENV = {'TIME_MAX': 10, 'STEPS': 251, 'omega0_per': 0.60,
              'K1': '10', 'K11': '10', 'K2': '100', 'K21': '10', 'K3': '0.5'}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': True}

PARAMS_TRAIN_GPS = {'UPDATES': 3, 'N': 1, 'M': 50,
                    'SHOW': True, 'rollouts': 100, 'samples': 2}

PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '4', '$y$': '4', '$z$': '4',
              '$p$': '0.0', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/34', r'$\theta$': 'np.pi/64',
              '$\phi$': 'np.pi/64'}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
