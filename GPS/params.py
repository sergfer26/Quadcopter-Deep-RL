PARAMS_LQG = {
    'alphas': '0.5**np.arange(8)',  # '1.1**(-np.arange(10)**2)',
    'delta_0': 2.0,
    'tol': 1e-6,
    'n_iterations': 100,
    'is_stochastic': False,
    'cov_reg': 3e-1
}
PARAMS_OFFLINE = {
    'lamb': 1e-3,
    'nu': 1e-1,
    'kl_step': 20,
    'min_eta': 1,
    'max_eta': 1e5,
    'rtol': 1e-1,
    'kl_maxiter': 1
}

PARAMS_ONLINE = {
    'step_size': 50,
    'F': 1e-4,
    'cov_reg': .2
}
