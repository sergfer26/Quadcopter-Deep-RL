PARAMS_ENV = {'TIME_MAX': 30, 'STEPS': 800, 'omega0_per': 0.60, 'FLAG':False} #Si es false los vuelos pueden terminar

PARAMS_TRAIN_DDPG = {'BATCH_SIZE': 128, 'EPISODES': 10000, 'SHOW': False}

PARAMS_TRAIN_SUPER = {'BATCH_SIZE': 64, 'EPOCHS': 1000,'N' : 1500, 'SHOW': False} #N es el numero de vuelos hechos con el control lineal
