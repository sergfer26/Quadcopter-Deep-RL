PARAMS_LQG = {
    'min_reg': 0.001,
    'max_reg': 1e4,
    'reg': 20,
    'alphas': '0.5**np.arange(8)',  # '1.1**(-np.arange(10)**2)',
    'delta_0': 2.0,
    'tol': 1e-6,
    'n_iterations': 100,
    'is_stochastic': True,
    'is_constrained': False,
    'cov_reg': 3e-1
}
PARAMS_OFFLINE = {
    'kl_step': 10,
    'min_eta': 1e-4,
    'max_eta': 1e3,
    'rtol': 1e-2,
    'kl_maxiter': 1
}

PARAMS_ONLINE = {
    'step_size': 50,
    'F': 1e-4
}
