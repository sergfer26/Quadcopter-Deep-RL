
PARAMS_ENV = {'TIME_MAX': 10, 'STEPS': 251, 'omega0_per': 0.60,
              'K1': '10', 'K11': '10', 'K2': '100', 'K21': '10', 'K3': '.5'}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': True}

PARAMS_TRAIN_GPS = {'UPDATES': 5, 'N': 3, 'M': 100,
                    'SHOW': True, 'is_stochastic': False, 'samples': 3,
                    'batch_size': 128}

PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '2', '$y$': '2', '$z$': '2',
              '$p$': '0.0', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/16', r'$\theta$': 'np.pi/16',
              '$\phi$': 'np.pi/16'}

PARAMS_DDPG = {'hidden_sizes': [64, 64], 'actor_learning_rate': '1e-3',
               'critic_learning_rate': 1e-4, 'gamma': 0.98, 'tau': 0.125,
               'max_memory_size': int(1e4)}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
