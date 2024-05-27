import numpy as np
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt
from params import STATE_NAMES
from tqdm import tqdm


labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
          ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
          ('$r$', '$\psi$')
          ]


def classifier(state, goal_state=None, c=5e-1, mask: np.ndarray = None):
    if not isinstance(goal_state, np.ndarray):
        goal_state = np.zeros_like(state)
    return np.apply_along_axis(criterion, 0, state, goal_state, c, mask).all()


def criterion(x, y=0, c=5e-1, mask: np.ndarray = None):
    if isinstance(mask, np.ndarray):
        x = x[mask]
        y = y[mask]
    return abs(x - y) < c


def plot_classifier(states, cluster, x_label='x', y_label='y',
                    figsize=(6, 6), dpi=300, ax=None,
                    style="seaborn-whitegrid"):
    cmap = None
    plt.style.use(style)
    if not isinstance(ax, plt.Axes):
        ax = plt.subplots(figsize=figsize, dpi=dpi)[1]
    if cluster.all():
        cluster = 'blue'
    else:
        cmap = mpl.colors.ListedColormap(['red', 'blue'])

    sc = ax.scatter(states[0], states[1], c=cluster, s=10, alpha=0.2,
                    cmap=cmap)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax, sc


def confidence_region(states, goal_states=None, c=5e-1, mask: np.ndarray = None):
    if not isinstance(goal_states, np.ndarray):
        goal_states = np.zeros_like(states)
    return np.apply_along_axis(classifier, -1, states, goal_states, c, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-array', type=str,
                        default='results_gps/23_07_31_12_15/rollouts/23_08_26_23_40/policy/states_60.npz')
    parser.add_argument('--list-steps', nargs='+', type=int,
                        help='List of seconds of interest',
                        default=[125, 250, 750, 1500]
                        )
    parser.add_argument('--style', default='seaborn-v0_8-whitegrid', type=str)
    parser.add_argument('--threshold', default=1, type=float)
    args = parser.parse_args()

    dt = 0.04
    policy_path = '/'.join(args.file_array.split('/')[:-1])

    # (6, 10000, 1501, 12) (n_x //2, samples, T, n_x)
    states = np.load(args.file_array)['states']
    init_states = states[:, :, 0]
    # steps=int(t * env.dt))

    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    state_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
    for t in tqdm(args.list_steps, desc='Rendering image...', unit='image'):
        bool_state = confidence_region(
            states[:, :, int(t)],
            c=args.threshold,
            mask=state_mask
        )
        # cluster = np.apply_along_axis(get_color, -1, bool_state)
        fig, axes = plt.subplots(figsize=(14, 10), nrows=len(labels)//3,
                                 ncols=3, dpi=250, sharey=False)
        axs = axes.flatten()
        for i in range(init_states.shape[0]):
            mask = abs(init_states[i, 0]) > 0
            label = np.array(STATE_NAMES)[mask]
            plot_classifier(
                init_states[i, :, mask],
                bool_state[i], x_label=label[0],
                y_label=label[1],
                ax=axs[i],
                style=args.style)
        fig.suptitle(f'Política, tiempo: {t * dt}')
        th = str(args.threshold).replace('.', '_')
        fig.savefig(policy_path + f'samples-{int(t * dt)}-th_{th}.png')
