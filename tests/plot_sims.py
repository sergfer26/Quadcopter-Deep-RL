from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import argparse


high = np.array([
    # u, v, w, x, y, z, p, q, r, psi, theta, phi
    [20., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 20., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 20., 0., 0., 10., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., .0, 0., 0., 0., np.pi/2],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., np.pi/2, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., np.pi/2, 0., 0.]
])

low = -high
STATE_SPACE = np.stack([low, high])
PARAMS_OBS = {'$u$': '0.0', '$v$': '0.0', '$w$': '0.0',
              '$x$': '8', '$y$': '8', '$z$': '8',
              '$p$': '0.00', '$q$': '0.0', '$r$': '0.0',
              '$\psi$': 'np.pi/32', r'$\theta$': 'np.pi/32',
              '$\\varphi$': 'np.pi/32'}

# Etiquetas
STATE_NAMES = list(PARAMS_OBS.keys())


labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
          ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
          ('$r$', '$\psi$')
          ]


def plot_classifier(states, cluster, x_label='x', y_label='y',
                    figsize=(6, 6), dpi=300, ax=None):
    cmap = None
    if not isinstance(ax, plt.Axes):
        ax = plt.subplots(figsize=figsize, dpi=dpi)[1]
    if cluster.all():
        cluster = 'blue'
    else:
        cmap = mpl.colors.ListedColormap(['red', 'blue'])

    # sc = ax.scatter(states[0], states[1], c=cluster, s=10, alpha=0.2,
    #                cmap=cmap)
    x_class0 = states[0, ~cluster]
    y_class0 = states[1, ~cluster]

    x_class1 = states[0, cluster]
    y_class1 = states[1, cluster]

    sc = ax.scatter(x_class0, y_class0, color='red',
                    alpha=0.3)  # Lower alpha for class 0
    sc = ax.scatter(x_class1, y_class1, color='blue',
                    alpha=0.7)   # Full alpha for class 1

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax, sc


def classifier(state, goal_state=None, c=5e-1, mask: np.ndarray = None):
    if not isinstance(goal_state, np.ndarray):
        goal_state = np.zeros_like(state)
    return np.apply_along_axis(criterion, 0, state, goal_state, c, mask).all()


def criterion(x, y=0, c=5e-1, mask: np.ndarray = None):
    if isinstance(mask, np.ndarray):
        x = x[mask]
        y = y[mask]
    return abs(x - y) < c


def confidence_region(states, goal_states=None, c=5e-1, mask: np.ndarray = None):
    if not isinstance(goal_states, np.ndarray):
        goal_states = np.zeros_like(states)
    return np.apply_along_axis(classifier, -1, states, goal_states, c, mask)


def get_color(bools):
    return np.array(['b' if b else 'r' for b in bools])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='distance')
    parser.add_argument('--file-array', type=str, default=None)
    parser.add_argument('--times', nargs='+', type=int,
                        help='A list of values', default=[15, 30, 60])
    parser.add_argument('--one-figure', action='store_true',
                        default=False, help='Enable saving one figure')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    # path = "results_ilqr/stability_analysis/23_07_14_11_30/stability_region.npz"

    if isinstance(args.file_array, str):
        path = args.file_array
    else:
        path = "results_gps/23_04_22_02_26/rollouts/24_05_26_21_25/policystates_60.npz"

    save_path = '/'.join(path.split('/')[:-1])
    array = np.load(path)
    states = array['states']
    num_experiments = states.shape[0]
    th = args.threshold

    init_states = states[:, :, 0]
    state_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for t in args.times:
        index = int(t * 25.00) + 1
        print('Getting confidence region...')
        bool_state = confidence_region(
            states[:, :, t],
            c=th,
            mask=state_mask
        )
        print('Confidence region setted...')

        mask1 = np.apply_along_axis(lambda x, y: np.greater(
            abs(x), y), -1, states[:, 0, 0], 0)
        mask2 = STATE_SPACE[1] > 0
        mask2 = mask2[-num_experiments:]
        indices = np.array([
            np.where(np.all(mask1 == mask2[i], axis=1))[0] for i in range(num_experiments)
        ]).squeeze()
        states = states[indices]
        init_states = states[:, :, 0]
        style = "seaborn-v0_8-whitegrid"
        plt.style.use(style)

        th_str = str(th).replace('.', '_')

        if args.one_figure:
            fig, axes = plt.subplots(dpi=250)

        step = args.times.index(t) + 1
        for i in tqdm(range(init_states.shape[0]), desc=f"step {t}/{len(args.times)}"):
            if args.one_figure:
                ax = axes.flatten()[i]
            else:
                fig, ax = plt.subplots(dpi=250)
            mask = abs(init_states[i, 0]) > 0
            label = np.array(STATE_NAMES)[mask]
            plot_classifier(init_states[i, :, mask],
                            bool_state[i], x_label=label[0],
                            y_label=label[1], ax=ax
                            )
            plt.tight_layout()

            if not args.one_figure:
                file_path = f'{save_path}/stability_{label[0]}-{label[1]}_th-{th_str}_t-{t}.png'.replace(
                    '$', '').replace('\\', '')
                fig.savefig(file_path)
                print(f'  ==> file {file_path} saved.')

        if args.one_figure:
            file_path = f'{save_path}/stability_state_th-{th_str}_t{t}.png'
            fig.savefig(file_path)
            print(f'  ==> file {file_path} saved.')
