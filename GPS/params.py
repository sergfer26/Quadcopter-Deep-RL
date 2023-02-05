PARAMS_iLQR = {
    'min_reg': 0.001,
    'max_reg': 1e4,
    'reg': 20,
    'alphas': '0.5**np.arange(8)',  # '1.1**(-np.arange(10)**2)',
    'delta_0': 2.0,
    'tol': 1e-6,
    'n_iterations': 100,
    'q1': 0.5,
    'q2': 0.01,
    'is_stochastic': True,
    'horizon': 5,
    'kl_step': 6
}
