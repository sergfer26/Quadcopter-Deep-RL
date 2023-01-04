
PARAMS_ENV = {'TIME_MAX': 1/5, 'STEPS': 5, 'omega0_per': 0.60, 'FLAG': False,
              'K1': 0.25, 'K2': 0.1, 'K3': 0.005}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': False}

PARAMS_TRAIN_PPO = {'EPISODES': 1000, 'n': 10, 'SHOW': True,
                    'action_std_decay_freq': int(630)}

PARAMS_TRAIN_GCL = {'REWARD_UPDATES': 2, 'DEMO_SIZE': 64, 'SAMP_SIZE': 128,
                    'n': 2, 'SHOW': False}

PARAMS_OBS = {'$u$': '0.1', '$v$': '0.1', '$w$': '0.1',
              '$x$': '1.0', '$y$': '1.0', '$z$': '1.0',
              '$p$': '0.1', '$q$': '0.1', '$r$': '0.1',
              '$\psi$': '0.1', r'$\theta$': '0.1',
              '$\phi$': '0.1'}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']

COST_NAMES = ['$c_t$', r'$\sum c_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
