PARAMS_ENV = {'TIME_MAX': 5, 'STEPS': 130, 'omega0_per': 0.60, 'FLAG':False} #Si es false los vuelos pueden terminar

PARAMS_TRAIN = {'BATCH_SIZE': 128, 'EPISODES': 20, 'SHOW': False}

PARAMS_TRAIN_SUPER = {'BATCH_SIZE': 64, 'EPOCHS': 5,'N' : 5, 'SHOW': False} #N es el numero de vuelos hechos con el control lineal