import numpy as np
import warnings
from scipy.stats import multivariate_normal
from scipy.optimize import brentq
from ilqr import iLQR
from .params import PARAMS_LQG
from .utils import mvn_kl_div, OnlineCost, OfflineCost, nearestPD


class iLQG(iLQR):

    # env: gym.Env, dynamics: Dynamics, cost: Cost):
    def __init__(self, dynamics,
                 cost, steps: int,
                 min_reg=1e-3,
                 max_reg=1e4,
                 reg=20,
                 delta_0=2.0,
                 is_stochastic=PARAMS_LQG['is_stochastic']
                 ):
        '''
        dynamics : `ilqr.Dynamics`
            Es el sistema dinámico bajo el cual funciona el sistema.
            Para más detalles, revisar containers.Dynamics.
        cost : `ilqr.Cost`
            Es el costo asociado a los objetivos del agente, depende
            del estado del sistema y los valores del control.
        steps : int
            Es el número de transiciones que hay en un episodio.
        low : `np.ndarray`
            Es el límite inferior del espacio del estado.
        high : `np.ndarray`
            Es el límite superior del espacio del estado.
        '''
        super(iLQG, self).__init__(dynamics, cost, steps, max_reg, False)
        # time step index for simulation
        self.i = 0

        # Regularization parameters
        self._mu_min = min_reg
        self._mu = reg
        self._delta_0 = delta_0
        self.alpha = 1.0

        # Dynamics characteristics
        self.num_states = dynamics._state_size
        self.num_actions = dynamics._action_size

        # Control parameters
        self._C = np.stack([np.identity(self.num_actions)
                           for _ in range(self.N)])
        self._nominal_us = np.empty((self.N, self.num_actions))
        self._nominal_xs = np.empty((self.N + 1, self.num_states))
        self._xs = np.empty_like(self._nominal_xs)
        self._us = np.empty_like(self._nominal_us)
        self.is_stochastic = is_stochastic

    def _rollout(self, x0, us):
        '''
        Rollout with initial state and control trajectory
        '''
        xs = np.empty((us.shape[0]+1, x0.shape[0]))
        xs[0] = x0
        cost = 0
        for n in range(us.shape[0]):
            xs[n+1] = self.dynamics.f(xs[n], us[n], n)
            cost += self.cost.l(xs[n], us[n], n)
        cost += self.cost.l(xs[-1], None, n+1, terminal=True)
        return xs, cost

    def fit_control(self, x0, us_init,
                    n_iterations=PARAMS_LQG['n_iterations'],
                    tol=PARAMS_LQG['tol'],
                    callback=True):
        '''
        Argumentos
        ----------
        x0 : `np.ndarray`
            Es un arreglo que representa el estado inicial del entrenamiento.
            `x0.shape -> self.num_states`.
        us_init : `np.ndarray`
            Es un arreglo que representa la trayectoria de controles no
            necesariamente optimos. Estos controles seran ajustados por
            el algoritmo ilqr. `us_init.shape -> (self.N, self.num_actions)`.
        maxiters : int
            Es el número máximo de iteraciones que debe usarse el método
            de corrección.
        early_stop : bool
            Determina si el algoritmo puede ser detenido antes del valor
            maxiters o no.

        Retornos
        --------
        xs : `np.ndarray`
            Es un arreglo que representa la trayectoria de estados.
            `xs.shape = (self.N + 1, self.num_states)`
        us : `np.ndarray`
            Es un arreglo que representa la trayectoria de controles óptimas.
            `us.shape = (self.N, self.num_actions)`
        cost_trace : list
            Evolución del costo.
        '''
        on_iteration = self.on_iteration if callback else None
        xs, us = self.fit(x0, us_init, n_iterations=n_iterations,
                          tol=tol, on_iteration=on_iteration)
        # Store fit parameters.
        cost_trace = self._step_cost(xs, us)

        return xs, us, cost_trace

    def get_action(self, state, update_time_step=True):
        action = self._nominal_us[self.i] + self.alpha * self._k[self.i] + \
            self._K[self.i] @ (state - self._nominal_xs[self.i])
        if self.is_stochastic:
            C = self._C[self.i]  # + PARAMS_LQG['cov_reg'] * \
            # np.identity(self.num_actions)
            action = multivariate_normal.rvs(action, C, 1)
        if update_time_step:
            self.i += 1
            self.i = self.i % (self.N)

        return action

    def rollout(self, x0):
        self.reset()
        us = np.empty_like(self._nominal_us)
        xs = np.empty_like(self._nominal_xs)
        xs[0] = x0
        for i in range(self.N):
            us[i] = self.get_action(xs[i])
            xs[i+1] = self.dynamics.f(xs[i], us[i], i)
        return xs, us

    def get_prob_action(self, state, action, t=0):
        g_xs = self._nominal_us[t] + self.alpha * self._k[t] + \
            self._K[t] @ (state - self._nominal_xs[t])
        cov = self._C[t] + PARAMS_LQG['cov_reg'] * \
            np.identity(self.num_actions)
        return multivariate_normal.pdf(x=action, mean=g_xs, cov=cov, allow_singular=True)

    def _step_cost(self, xs, us):
        J = map(lambda args: self.cost.l(*args),
                zip(xs[:-1], us, range(self.N)))
        J = list(J)
        J.append(self.cost.l(xs[-1], None, self.N, terminal=True))
        return J

    def reset(self):
        self.i = 0

    def on_iteration(self, iteration, x, us, J_opt, accepted, converged):
        x = x.tolist()
        us = us.tolist()
        text = f'— it= {iteration}, J= {J_opt}, accepted= {accepted}, converged= {converged}'
        print(text)

    def fit(self, x0, us_init, n_iterations=100, tol=1e-6, on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted
                    result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 20
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = eval(PARAMS_LQG['alphas'])  # 1.1**(-np.arange(10)**2)
        alpha = 1.0

        us = us_init.copy()
        xs_old = np.zeros_like(self._nominal_xs)
        us_old = np.zeros_like(self._nominal_us)
        # N = us.shape[0]
        k = self._k
        K = self._K
        C = self._C

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                 F_uu) = self._forward_rollout(x0, us)
                J_opt = L.sum()
                changed = False

            try:
                # Backward pass.
                k, K, C = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux,
                                              L_uu, F_xx, F_ux, F_uu)

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, C, alpha,
                                                   is_stochastic=self.is_stochastic)
                    J_new = self._trajectory_cost(xs_new, us_new)
                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True

                        J_opt = J_new
                        xs_old = xs
                        us_old = us
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break

            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))
                # reg = self._mu * np.eye(self.dynamics.state_size)
                # Q_uu = L_uu + F_u.T.dot(V_xx + reg).dot(F_u)

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._C = C
        self._xs = xs_old
        self._us = us_old
        self._nominal_xs = xs
        self._nominal_us = us
        self.alpha = alpha

        return xs, us

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """ Computes the feedforward and feedback gains k and K.
        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].
        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]
        N = F_x.shape[0]
        k = np.empty((N, self.num_actions))
        K = np.empty((N, self.num_actions, self.num_states))
        C = np.empty((N, self.num_actions, self.num_actions))

        for i in range(N - 1, -1, -1):
            if self._use_hessians:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx,
                                                     F_xx[i], F_ux[i], F_uu[i])
            else:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx)
            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)
            C[i] = np.linalg.inv(Q_uu)
            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])
            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.
        return k, K, C

    def _control(self, xs, us, k, K, C, alpha=1.0,
                 is_stochastic=True, x0=None):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        N = us.shape[0]
        if isinstance(x0, np.ndarray):
            xs_new[0] = x0
        else:
            xs_new[0] = xs[0].copy()

        for i in range(N):
            # Eq (12).
            us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])
            if is_stochastic:
                cov = C[i]
                us_new[i] = multivariate_normal.rvs(us_new[i], cov, 1)

            # Eq (8c).
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i)

        return xs_new, us_new

    def save(self, path, file_name='ilqr_control.npz'):
        file_path = path + file_name
        np.savez(file_path,
                 C=self._C,
                 K=self._K,
                 k=self._k,
                 xs=self._nominal_xs,
                 us=self._nominal_us,
                 xs_old=self._xs,
                 us_old=self._us,
                 alpha=self.alpha
                 )

    def load(self, path, file_name='ilqr_control.npz'):
        file_path = path + file_name
        npzfile = np.load(file_path)
        self._k = npzfile['k']
        self._K = npzfile['K']
        self._C = npzfile['C']
        self._nominal_xs = npzfile['xs']
        self._nominal_us = npzfile['us']
        self.alpha = npzfile['alpha']


class OfflineController(iLQG):

    def __init__(self, dynamics,
                 cost: OfflineCost, steps: int,
                 min_reg=1e-3,
                 max_reg=1e4,
                 reg=20,
                 delta_0=2.0,
                 is_stochastic=PARAMS_LQG['is_stochastic'],
                 known_dynamics=True):
        super().__init__(dynamics, cost, steps, min_reg, max_reg,
                         reg, delta_0, is_stochastic)
        # Cost regularization parametrs
        self.check_constrain = False
        self.known_dynamics = known_dynamics

    def save(self, path, file_name='ilqr_control.npz'):
        file_path = path + file_name
        C = np.array([nearestPD(self._C[i]) for i in range(self._C.shape[0])])
        C += PARAMS_LQG['cov_reg'] * np.identity(self._C.shape[-1])
        np.savez(file_path,
                 C=C,
                 K=self._K,
                 k=self._k,
                 xs=self._nominal_xs,
                 us=self._nominal_us,
                 xs_old=self._xs,
                 us_old=self._us,
                 alpha=self.alpha,
                 eta=self.cost.eta
                 )

    def load(self, path, file_name='ilqr_control.npz'):
        file_path = path + file_name
        npzfile = np.load(file_path)
        self._k = npzfile['k']
        self._K = npzfile['K']
        self._C = np.round(npzfile['C'], 5)
        self._nominal_xs = npzfile['xs']
        self._nominal_us = npzfile['us']
        self.alpha = npzfile['alpha']
        if isinstance(self.cost, OfflineCost):
            self.cost.eta = npzfile['eta']

    def step(self, eta: float):
        us = self.us_init
        N = us.shape[0]
        self.cost.eta = eta
        params = self.cost.control_parameters()
        params['is_stochastic'] = False
        params['x0'] = self.x0

        (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
         F_uu) = self._forward_rollout(self.x0, us)

        k, K, C = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                      F_xx, F_ux, F_uu)

        us_new = self._control(
            xs, us, k, K, C, params['alpha'], False, x0=self.x0)[1]
        us_old = self._control(**params)[1]
        C_old = params['C']
        C_new = np.array([nearestPD(C[i]) for i in range(N)])
        C_new += PARAMS_LQG['cov_reg'] * np.identity(us.shape[-1])
        kl_div = sum([mvn_kl_div(us_new[j], us_old[j], C_new[j], C_old[j])
                      for j in range(N)])
        return kl_div

    def _backward_pass(self, F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                       F_xx=None, F_ux=None, F_uu=None):
        k, K, C = super()._backward_pass(F_x, F_u, L_x, L_u,
                                         L_xx, L_ux, L_uu, F_xx, F_ux, F_uu)
        C = np.array([nearestPD(C[j]) for j in range(C.shape[0])])
        return k, K, C

    def optimize(self, kl_step: float,
                 min_eta: float = 1e-4,
                 max_eta: float = 1e4,
                 rtol: float = 1e-1, kl_maxiter=2):
        """Perform iLQG trajectory optimization.
        Args:
            kl_step: KL divergence threshold to previous policy
            min_eta: Minimal value of the Lagrangian multiplier
            max_eta: Maximal value of the Lagrangian multiplier
            rtol: Tolerance of found solution to kl_step. Levine et al.
            propose a value of 0.1 in "Learning Neural Network Policies with
                  Guided Policy Search under Unknown Dynamics", chapter 3.1
            full_history: Whether to return ahistory of all optimization
            steps, for debug purposes
        Returns:
            result: A `ILQRStepResult` or
                    a list of `ILQRStepResult` if `full_history` is enabled
                    (in order they were visited)
        """
        if self.known_dynamics:
            # Check if constraind is fulfilled at maximum deviation
            if self.step(min_eta) <= kl_step:
                # return self.step(min_eta)
                self.cost.eta = min_eta
            else:
                # Check if constraint cen be fulfilled at all
                if self.check_constrain:
                    kl_div = self.step(max_eta)
                    if kl_div > kl_step:
                        raise ValueError(
                            f"max_eta eta to low: ({max_eta}), kl_div: {kl_div}")

                # Find the point where kl divergence equals the kl_step
                def constraint_violation(log_eta):
                    return self.step(np.exp(log_eta)) - kl_step

                # Search root of the constraint violation
                # Perform search in log-space, as this requires much fewer
                # iterations
                print("Brent's method begins...")
                log_eta = brentq(
                    constraint_violation, np.log(min_eta), np.log(max_eta),
                    rtol=rtol, maxiter=kl_maxiter, disp=False)

                print(f"eta= {np.exp(log_eta)}")
                self.cost.eta = np.exp(log_eta)

        print("iLQR optimization step...")
        xs, us = self.fit(self.x0, self.us_init,
                          n_iterations=PARAMS_LQG['n_iterations'],
                          tol=PARAMS_LQG['tol'],
                          on_iteration=self.on_iteration)
        cost_trace = self._step_cost(xs, us)
        params = self.cost.control_parameters()
        params['is_stochastic'] = False
        us_old = self._control(**params)[1]
        us_new = self._control(xs, us, self._k, self._K,
                               self._C, self.alpha, False)[1]
        N = us.shape[0]
        C_old = params['C']
        C_new = np.array([nearestPD(self._C[i]) for i in range(N)])
        C_new += PARAMS_LQG['cov_reg'] * np.identity(us.shape[-1])
        kl_div = sum([mvn_kl_div(us_new[j], us_old[j], C_new[j], C_old[j])
                      for j in range(N)])
        return xs, us, cost_trace, kl_div


class OnlineController(iLQG):

    def __init__(self, dynamics, cost: OnlineCost,
                 steps: int,
                 min_reg=1e-3,
                 max_reg=1e4,
                 reg=20,
                 delta_0=2.0,
                 is_stochastic=PARAMS_LQG['is_stochastic']):
        super().__init__(dynamics, cost, steps, min_reg, max_reg,
                         reg, delta_0, is_stochastic)

    def fit(self, x0, us_init, n_iterations=100, tol=1e-1,
            on_iteration=None):
        xs, us = self.cost.control.rollout(x0)
        # update the cost parameters
        self.cost._dist_dynamics(xs, us)
        return super().fit(x0, us_init, n_iterations, tol, on_iteration)


class OnlineMPC(object):

    """Receding horizon controller for Model Predictive Control."""

    def __init__(self, x0, controller: OnlineController):
        """Constructs a RecedingHorizonController.

        Args:
            x0: Initial state [state_size].
            controller: Controller to fit with.
        """
        self._x = x0
        self._controller = controller
        self._random = np.random.RandomState()

    def seed(self, seed):
        self._random.seed(seed)

    def set_state(self, x):
        """Sets the current state of the controller.

        Args:
            x: Current state [state_size].
        """
        self._x = x

    def control(self,
                us_init,
                step_size=1,
                initial_n_iterations=100,
                subsequent_n_iterations=1,
                *args,
                **kwargs):
        """Yields the optimal controls to run at every step as a receding
        horizon problem.

        Note: The first iteration will be slow, but the successive ones will be
        significantly faster.

        Note: This will automatically move the current controller's state to
        what the dynamics model believes will be the next state after applying
        the entire control path computed. Should you want to correct this state
        between iterations, simply use the `set_state()` method.

        Note: If your cost or dynamics are time dependent, then you might need
        to shift their internal state accordingly.

        Args:
            us_init: Initial control path [N, action_size].
            step_size: Number of steps between each controller fit. Default: 1.
                i.e. re-fit at every time step. You might need to increase this
                depending on how powerful your machine is in order to run this
                in real-time.
            initial_n_iterations: Initial max number of iterations to fit.
                Default: 100.
            subsequent_n_iterations: Subsequent max number of iterations to
                fit. Default: 1.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to `controller.fit()`.

        Yields:
            Tuple of
                xs: optimal state path [step_size+1, state_size].
                us: optimal control path [step_size, action_size].
        """
        n_iterations = initial_n_iterations
        while True:
            xs, us = self._controller.fit(self._x,
                                          us_init,
                                          n_iterations=n_iterations,
                                          *args,
                                          **kwargs)
            self._x = xs[step_size]
            yield xs[:step_size + 1], us[:step_size]

            # Set up next action path seed by simply moving along the current
            # optimal path and appending random unoptimal values at the end.
            us_start = us[step_size:]
            # self._random.uniform(-1, 1, (step_size, action_size))
            _, us_end = self._controller.cost.control.rollout(self._x)
            us_end = us_end[-step_size:]
            us_init = np.vstack([us_start, us_end])
            n_iterations = subsequent_n_iterations
