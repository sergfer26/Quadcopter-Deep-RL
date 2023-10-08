
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PARAMS_TRAIN_RMDDPG = {'BATCH_SIZE': 256,
                       'EPISODES': 5, 'n': 2, 'SHOW': True,
                       'reset_noise': True,
                       'behavior_policy': 'ilqr',  # 'ilqr' # None
                       'behavior_path': 'models/ilqr_control_750.npz'  # 'results_gps/23_07_31_12_15/'
                       }

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 256,
                     'EPISODES': 1000, 'n': 100, 'SHOW': False,
                     'behavior_policy': True,
                     'behavior_path': 'results_gps/23_07_31_12_15/',
                     'pretrained': True
                     }


PARAMS_DDPG = {'hidden_sizes': [128, 128], 'actor_learning_rate': '1e-3',
               'critic_learning_rate': 3e-3, 'gamma': 0.99, 'tau': 2e-3,
               'max_memory_size': int(1e4)
               }


PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '8', '$y$': '8', '$z$': '8',
              '$p$': '0.00', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/32', r'$\theta$': 'np.pi/32',
              '$\\varphi$': 'np.pi/32'}

WEIGHTS = dict(u=1e-2, v=1e-2, w=1e-2,
               x=1e-3, y=1e-3, z=1e-3,
               p=2e-2, q=2e-2, r=1e-2,
               psi=5e-2, theta=1e-1, phi=1e-1
               )


PARAMS_ENV = {'dt': 0.04, 'STEPS': 750, 'omega0_per': 0.60,
              'K1': '10', 'K11': '10', 'K2': '100', 'K21': '10', 'K3': '.5'}
# Si es false los vuelos pueden terminar


PARAMS_TRAIN_GPS = {'UPDATES': 5, 'N': 7, 'M': 800,
                    'SHOW': True, 'is_stochastic': False, 'samples': 3,
                    'batch_size': 20, 'shuffle_batches': True, 'time_step': 1,
                    'policy_updates': 3}


# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())

ACTION_NAMES = [f'$\\omega_{i}$' for i in range(1, 5)]

REWARD_NAMES = ['$r_t$', r'$\sum r_t$']

high = np.array([
    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    [10., 0., 0., 20., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 10., 0., 0., 20., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 12., 0., 0., 25., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., .0, 0., 0., 0., np.pi/2],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., np.pi/2, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., np.pi/2, 0., 0.]
])

low = -high
state_space = np.stack([low, high])


# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal

# Define colors for the colormap
colors = ['royalblue', 'royalblue',
          'mediumpurple', 'mediumpurple',
          'orchid', 'orchid',
          'royalblue', 'royalblue',
          'mediumpurple', 'mediumpurple',
          'orchid', 'orchid'
          ]

# Create a custom colormap
state_cmap = LinearSegmentedColormap.from_list('CustomColormap', colors, N=12)

# Set ylimists for state variables
