PARAMS_LQG = {
    'min_reg': 0.001,
    'max_reg': 1e4,
    'reg': 20,
    'alphas': '0.5**np.arange(8)',  # '1.1**(-np.arange(10)**2)',
    'delta_0': 2.0,
    'tol': 1e-6,
    'n_iterations': 100,
    'is_stochastic': True,
    'cov_reg': .15
}
PARAMS_OFFLINE = {
    'kl_step': 6,
    'min_eta': 1e-4,
    'max_eta': 1e4,
    'rtol': 1e-1,
    'kl_maxiter': 1
}

PARAMS_ONLINE = {
    'step_size': 50,
    'F': 1e-4,
    'cov_reg': .2
}
