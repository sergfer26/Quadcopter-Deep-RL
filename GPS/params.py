PARAMS_LQG = {
    'alphas': '0.5**np.arange(8)',  # '1.1**(-np.arange(10)**2)',
    'delta_0': 2.0,
    'tol': 1e-1,
    'n_iterations': 4,
    'is_stochastic': False,
    'cov_reg': 1e-4
}
PARAMS_OFFLINE = {
    'lamb': '1e-12',
    'alpha_lamb': '1e-3',
    'nu': '1e-3',
    'kl_step': 800,
    'per_kl': .0,
    'adaptive_kl': False,
    'min_eta': '1',
    'max_eta': '1e8',
    'rtol': 1e-1,
    'kl_maxiter': 1
}

PARAMS_ONLINE = {
    'step_size': 50,
    'F': 1e-4,
    'cov_reg': .2
}
