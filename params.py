
PARAMS_ENV = {'dt': 0.04, 'STEPS': 750, 'omega0_per': 0.60,
              'K1': '10', 'K11': '10', 'K2': '100', 'K21': '10', 'K3': '.5'}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': True}

PARAMS_TRAIN_GPS = {'UPDATES': 5, 'N': 7, 'M': 800,
                    'SHOW': False, 'is_stochastic': False, 'samples': 3,
                    'batch_size': 20, 'shuffle_batches': True, 'time_step': 1,
                    'policy_updates': 2}

PARAMS_OBS = {'$u$': '0.5', '$v$': '0.5', '$w$': '0.5',
              '$x$': '5', '$y$': '5', '$z$': '5',
              '$p$': '0.1', '$q$': '0.1', '$r$': '0.1',
              '$\psi$': 'np.pi/64', r'$\theta$': 'np.pi/64',
              '$\phi$': 'np.pi/64'}

PARAMS_DDPG = {'hidden_sizes': [128, 128], 'actor_learning_rate': '1e-2',
               'critic_learning_rate': 1e-4, 'gamma': 0.98, 'tau': 0.125,
               'max_memory_size': int(1e4)}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
