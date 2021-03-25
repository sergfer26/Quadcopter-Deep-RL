PARAMS_DDPG = {'hidden_sizes': [64,64], 'actor_learning_rate': 1e-4, 
    'critic_learning_rate': 1e-4, 'gamma':0.98, 'tau':0.125, 'max_memory_size':int(1e7)}

PARAMS_UTILS = {'mu':0.0, 'theta': 0.15, 'max_sigma':0.5, 'min_sigma': 0.01, 'decay_period':800}