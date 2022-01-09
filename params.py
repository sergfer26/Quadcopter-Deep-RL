from numpy import pi

PARAMS_ENV = {'TIME_MAX': 5, 'STEPS': 124, 'omega0_per': 0.60, 'FLAG': False,
              'reward': 'r1', 'lamb': 0.5}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 1, 'n': 1, 'SHOW': False}

PARAMS_TRAIN_SUPER = {'BATCH_SIZE': 64,
                      'EPOCHS': 2, 'N': 200, 'n': 2, 'SHOW': False}

PARAMS_TRAIN_PPO = {'EPISODES': 1000, 'n': 10, 'SHOW': True,
                    'action_std_decay_freq': int(630)}

PARAMS_OBS = {'$u$': 0.0, '$v$': 0.0, '$w$': 0.0, '$x$': 1, '$y$': 1, '$z$': 1,
              '$p$': 0, '$q$': 0, '$r$': 0, '$\psi$': pi/18,
              r'$\theta$': pi/18, '$\phi$': pi/18}


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
