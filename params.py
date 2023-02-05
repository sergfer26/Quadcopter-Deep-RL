
PARAMS_ENV = {'TIME_MAX': 10, 'STEPS': 250, 'omega0_per': 0.60, 'FLAG': False,
              'K1': 0.25, 'K2': 0.1, 'K3': 0.005}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1000, 'n': 5, 'SHOW': False}

PARAMS_TRAIN_PPO = {'EPISODES': 1000, 'n': 10, 'SHOW': True,
                    'action_std_decay_freq': int(630)}

PARAMS_TRAIN_GCL = {'REWARD_UPDATES': 2, 'DEMO_SIZE': 64, 'SAMP_SIZE': 128,
                    'n': 2, 'SHOW': False}

PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '4.0', '$y$': '4.0', '$z$': '4.0',
              '$p$': '0.0', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/16', r'$\theta$': 'np.pi/16',
              '$\phi$': 'np.pi/16'}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$a_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']

COST_NAMES = ['$c_t$', r'$\sum c_t$']


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
