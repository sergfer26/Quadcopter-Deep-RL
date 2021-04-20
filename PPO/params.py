PARAMS_PPO = {'hidden_sizes': [64, 64], 'actor_learning_rate': 1e-3,
              'critic_learning_rate': 1e-4, 'gamma': 0.98, 'K_epochs': 128,
              'eps_clip': 0.125, 'action_std_init': 0.5,
              'action_std_decay_rate': 0.001, 'min_action_std': 0.001}
