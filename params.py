from numpy import pi

PARAMS_ENV = {'TIME_MAX': 30, 'STEPS': 800, 'omega0_per': 0.60, 'FLAG': False,
              'reward': 'r1', 'lamb': 0.5}
# Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128,
                     'EPISODES': 2000, 'n': 10, 'SHOW': False}

PARAMS_TRAIN_SUPER = {'BATCH_SIZE': 64,
                      'EPOCHS': 2, 'N': 200, 'n': 2, 'SHOW': False}

PARAMS_TRAIN_PPO = {'EPISODES': 2000, 'SHOW': False,
                    'action_std_decay_freq': int(630)}

PARAMS_OBS = {'$u$': 0.0, '$v$': 0.0, '$w$': 0.0, '$x$': 0, '$y$': 0, '$z$': 0.5,
              '$p$': 0, '$q$': 0, '$r$': 0, '$\psi$': pi/36,
              r'$\theta$': pi/36, '$\phi$': pi/36}

# https://docs.sympy.org/latest/modules/parsing.html
REWARDS = {'a': r'$r - c\|\textbf{x}\|$',
           'b': r'$\max(0, 1 - \|\textbf{x} \|) - c_1 \|R(\textbf{\theta})\| -c_2\|\textbf{\omega}\|$',
           'c': r'r - c_1\|\textbf{x}\| - c_2\|R(\textbf{\theta})\| - c_3\|\textbf{\omega}\|'}

# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
