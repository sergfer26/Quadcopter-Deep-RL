PARAMS_ENV = {'TIME_MAX': 5, 'STEPS': 130, 'omega0_per': 0.60, 'FLAG':False} #Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128, 'EPISODES': 5, 'n': 10, 'SHOW': False}

PARAMS_TRAIN_SUPER = {'BATCH_SIZE': 64, 'EPOCHS': 5, 'N': 1000, 'n': 10, 'SHOW': False} 
# N es el numero de vuelos hechos con el control lineal
# n es el numéro de vuelos de simulación de la red neuronal
