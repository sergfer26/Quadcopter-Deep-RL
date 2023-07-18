import pathlib
import numpy as np
import multiprocessing as mp
from gym import spaces
from env import QuadcopterEnv
from utils import date_as_path
from utils import plot_classifier
from send_email import send_email
from multiprocessing import Process
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from animation import create_animation
from GPS.controller import DummyController
from simulation import plot_rollouts, n_rollouts
from params import STATE_NAMES, ACTION_NAMES, REWARD_NAMES


def in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


class ConvexRegion(object):
    '''
    [1] https://es.wikipedia.org/wiki/Envolvente_convexa
    [2] https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    [3] https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull
    '''

    def __init__(self, region, eps, x0=None):
        self.x0 = x0
        self.hull = list()
        self.hull_path = list()
        self.set_convex_hull(region, eps)

    def set_convex_hull(self, region, eps):
        '''
        region : `np.ndarray`
            (n_x/2, sims, 2, n_x)
        '''
        bool_state = np.apply_along_axis(
            lambda x: np.linalg.norm(x) < eps, -1, region[:, :, -1])
        init_states = region[:, :, 0]
        self.indices = np.apply_along_axis(
            np.where, -1, init_states[:, 0]).flatten()
        for i in range(region.shape[0]):
            pos = init_states[i][bool_state[i]]
            self.hull.append(
                ConvexHull(pos[:, self.indices[2 * i: 2 * i + 2]])
            )
        self.min_bound = np.hstack(
            [self.hull[i].min_bound for i in range(region.shape[0])]
        )
        self.max_bound = np.hstack(
            [self.hull[i].max_bound for i in range(region.shape[0])]
        )

    def _sample_pair(self, i):
        while True:
            x = np.random.uniform(low=self.min_bound[2 * i: 2 * i + 2],
                                  high=self.max_bound[2 * i: 2 * i + 2])
            if in_hull(x, self.hull[i]):
                break
        return x

    def sample(self):
        aux = list()
        for i in range(self.min_bound.shape[0] // 2):
            aux.append(
                self._sample_pair(i)
            )
        return np.hstack(aux)


def rollout4mp(agent, env, mp_list, n=1, states_init=None):
    '''
    states : (n, env.steps, env.observation_space.shape[0])
    '''
    states = n_rollouts(agent, env, n=n, states_init=states_init)[0]
    mp_list.append(states)


def rollouts(agent, env, sims, state_space, x0=None, num_workers=None,
             inv_transform_x=None, transform_x=None):
    '''
    Retorno
    --------
    states : (np.ndarray)
        dimensión -> (state_space.shape[0], sims, env.steps,
                        env.observation_space.shape[0])
    '''
    if not isinstance(num_workers, int):
        num_workers = state_space.shape[1]

    if not isinstance(x0, np.ndarray):
        x0 = np.zeros_like(state_space[0, 0])

    states = mp.Manager().list()
    process_list = list()
    if hasattr(agent, 'env'):
        other_env = agent.env
    else:
        other_env = env
    init_states = np.empty((num_workers, sims, env.state.shape[0]))
    for i in range(num_workers):
        env.observation_space = spaces.Box(
            low=state_space[0, i]+x0, high=state_space[1, i]+x0,
            dtype=np.float64)
        init_states[i] = np.array(
            [env.observation_space.sample() for _ in range(sims)])
        init_state = init_states[i]
        if callable(transform_x):
            init_state = np.apply_along_axis(transform_x, -1, init_state)
        p = Process(target=rollout4mp, args=(
            agent, other_env, states, sims, init_state
        )
        )
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    states = np.array(list(states))  # (6, sims, T, n_x)
    # Ordenar indices
    aux = list(map(lambda i: (np.nonzero(states[i, 0, 0] - x0)[
               0][0], i), list(range(num_workers))))
    aux.sort(key=lambda x: x[0])
    indices = [x for _, x in aux]
    states = states[indices]
    if callable(inv_transform_x):
        states = np.apply_along_axis(inv_transform_x, -1, states)

    return states


def stability(agent, state_space, state_names, save_path,
              save_name='stability',
              eps=4e-1, with_x0=False, sims=int(1e4), convex_hull=False,
              n_cols=3, mask=None):
    plt.style.use("fivethirtyeight")
    env = QuadcopterEnv()
    x0 = np.zeros_like(env.state)
    if with_x0:
        x0 = agent._nominal_xs[0]
    states = rollouts(agent, env, sims, state_space, x0=x0)
    end_states = states[:, :, -1]
    if isinstance(mask, np.ndarray):
        end_states = end_states[:, :, mask]
    bool_state = np.apply_along_axis(
        lambda x: np.linalg.norm(x) < eps, -1, end_states)
    fig, axes = plt.subplots(
        figsize=(15, 10), nrows=state_space.shape[1]//n_cols, ncols=n_cols,
        dpi=300)
    axs = axes.flatten()
    init_states = states[:, :, 0]
    # sc = list()
    for i in range(init_states.shape[0]):
        mask = abs(init_states[i, 0] - x0) > 0
        label = np.array(state_names)[mask]
        plot_classifier(init_states[i, :, mask],
                        bool_state[i], x_label=label[0],
                        y_label=label[1], ax=axs[i],
                        )[1]
        if convex_hull:
            pos = init_states[i][bool_state[i]]
            hull = ConvexHull(pos[:, mask])
            for simplex in hull.simplices:
                axs[i].plot(hull.points[simplex, 0],
                            hull.points[simplex, 1], '-k')

    #    sc.append(aux)

    fig.suptitle(f'Control iLQR \n $\epsilon=${eps}, T={env.steps}')
    fig.savefig(save_path + save_name + '_'+f'{sims}.png')
    np.savez(
        save_path + save_name + '_'+f'{sims}.npz',
        states=states[:, :, [0, env.steps]],
        bounds=state_space[1]
    )
    return states[:, :, [0, env.steps]]


if __name__ == '__main__':
    control_type = 'linear'
    env = QuadcopterEnv()
    sims = int(1e1)
    T = env.steps

    if control_type == 'ilqr':
        from params import state_space as STATE_SPACE
        control_path = 'models/'
        PATH = 'results_ilqr/stability_analysis/'+date_as_path()+'/'
        # 1. Stability analysis
        agent = DummyController(control_path, f'ilqr_control_{T}.npz')
        n_cols = 3
        eps = 4e-1
        mask = None

    elif control_type == 'linear':
        from Linear.constants import state_space as STATE_SPACE
        from Linear.agent import LinearAgent
        PATH = 'results_linear/stability_analysis/' + date_as_path()+'/'
        agent = LinearAgent(env)
        n_cols = 2
        eps = 3e-1
        mask = np.array(
            [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype='bool'
        )

    pathlib.Path(
        PATH + 'sample_rollouts/').mkdir(parents=True, exist_ok=True)
    region = stability(agent, STATE_SPACE, STATE_NAMES,
                       PATH, eps=eps, sims=sims, n_cols=n_cols,
                       mask=mask)
    if control_type != 'linear':
        convex_region = ConvexRegion(region, eps=eps)

        # 2. Test over stable regions
        init_states = np.vstack([convex_region.sample() for i in range(100)])
        states, actions, scores = n_rollouts(
            agent, env, n=100, states_init=init_states)

        fig1, _ = plot_rollouts(states, env.time, STATE_NAMES, alpha=0.05)
        fig1.savefig(PATH + 'state_rollouts.png')
        fig2, _ = plot_rollouts(actions, env.time, ACTION_NAMES, alpha=0.05)
        fig2.savefig(PATH + 'action_rollouts.png')
        fig3, _ = plot_rollouts(scores, env.time, REWARD_NAMES, alpha=0.05)
        fig3.savefig(PATH + 'score_rollouts.png')

        # 3. Samples' animation
        sample_indices = np.random.randint(states.shape[0], size=3)
        states_samples = states[sample_indices]
        actions_samples = actions[sample_indices]
        scores_samples = scores[sample_indices]
        create_animation(states_samples, actions_samples, env.time,
                         scores=scores_samples,
                         state_labels=STATE_NAMES,
                         action_labels=ACTION_NAMES,
                         score_labels=REWARD_NAMES,
                         file_name='flight',
                         path=PATH + 'sample_rollouts/')

    send_email(credentials_path='credentials.txt',
               subject='Termino de analisis de estabilidad: ' + PATH,
               reciever='sfernandezm97@ciencias.unam.mx',
               message=f'T={T} \n eps={eps} \n sims={sims}',
               path2images=PATH
               )
