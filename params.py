from numpy import pi

PARAMS_ENV = {'TIME_MAX': 10, 'STEPS': 260, 'omega0_per': 0.60, 'FLAG': False,
              'reward': 'r1'}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 2, 'n': 1, 'SHOW': False}

PARAMS_TRAIN_SUPER = {'BATCH_SIZE': 64,
                      'EPOCHS': 2, 'N': 200, 'n': 2, 'SHOW': False}

PARAMS_TRAIN_PPO = {'EPISODES': 15000, 'SHOW': False,
                    'action_std_decay_freq': int(630)}

PARAMS_OBS = {'$u$': 1, '$v$': 1, '$w$': 1, '$x$': 20, '$y$': 20, '$z$': 20, '$p$': 1,
              '$q$': 1, '$r$': 1, '$\psi$': pi/4, r'$\theta$': pi/4, '$\phi$': pi/4}

# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
