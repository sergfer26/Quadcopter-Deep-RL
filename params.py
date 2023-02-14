
PARAMS_ENV = {'TIME_MAX': 2, 'STEPS': 51, 'omega0_per': 0.60,
              'K1': .25, 'K11': .01, 'K2': .1, 'K21': .01, 'K3': .005}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': False}

PARAMS_TRAIN_GPS = {'UPDATES': 2, 'N': 3, 'M': 2, 'SHOW': False, 'rollouts': 2}

PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '2', '$y$': '2', '$z$': '2',
              '$p$': '0.0', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/16', r'$\theta$': 'np.pi/16',
              '$\phi$': 'np.pi/16'}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
