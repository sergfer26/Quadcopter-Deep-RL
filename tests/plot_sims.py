from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import argparse
from typing import Union, Tuple


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


def plot_classifier(states, cluster, x_label: str = 'x', y_label: str = 'y',
                    figsize=(6, 6), dpi=300, ax=None):
    cmap = None
    if not isinstance(ax, plt.Axes):
        ax = plt.subplots(figsize=figsize, dpi=dpi)[1]
    if cluster.all():
        cluster = 'blue'
    else:
        cmap = mpl.colors.ListedColormap(['red', 'blue'])
    sc = ax.scatter(states[0], states[1], c=cluster, s=10, alpha=0.3,
                    cmap=cmap)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    return ax, sc


def classifier(state: np.ndarray, c: float = 5e-1, mask: np.ndarray = None,
               ord: Union[int, str] = 2) -> np.ndarray:
    '''
    ord : {int, str}
    '''
    ord = int(ord) if ord.isdigit() else np.inf
    if isinstance(mask, np.ndarray):
        state = state[mask]
    return np.linalg.norm(state, ord=ord) < c


def confidence_region(states: np.ndarray, c: float = 5e-1, mask: np.ndarray = None,
                      ord: Union[int, float, str] = 2) -> np.ndarray:
    '''
    ord : {int, str: inf}
    '''
    return np.apply_along_axis(classifier, -1, states, c, mask, ord)


def get_color(bools):
    return np.array(['b' if b else 'r' for b in bools])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-array', type=str, default=None)
    parser.add_argument('--times', nargs='+', type=int,
                        help='A list of values', default=[15, 30, 60])
    parser.add_argument('--one-figure', action='store_true',
                        default=False, help='Enable saving one figure')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--ord', type=str, default='2')
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
    for t in tqdm(args.times):
        index = int(t * 25.00) + 1
        print(f'Getting confidence region at {t} seconds...')
        bool_state = confidence_region(
            states[:, :, t],
            c=th,
            mask=state_mask,
            ord=args.ord
        )
        # for i, label in enumerate(labels):
        stability_rate_total = np.sum(bool_state, axis=(
            0, 1)) / (bool_state.shape[-1] * bool_state.shape[0])
        print(f" ==> total stability rate: {stability_rate_total:.2f}")

        stability_rate_axis = np.sum(bool_state, axis=1) / bool_state.shape[-1]
        for j, (l_x, l_y) in enumerate(labels):
            print(
                f"  ==> {l_x}-{l_y} stability rate: {stability_rate_axis[j]}"
            )

            # The mask for the current pertubation positions
        mask1 = np.apply_along_axis(lambda x, y: np.greater(
            abs(x), y), -1, states[:, 0, 0], 0)

        # The mask for the number of pertubation settings
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
            fig, axes = plt.subplots(
                dpi=300, nrows=2, ncols=init_states.shape[0] // 2, figsize=(20, 10))

        step = args.times.index(t) + 1
        for i in range(init_states.shape[0]):
            if args.one_figure:
                ax = axes.flatten()[i]
            else:
                fig, ax = plt.subplots(dpi=300)
            mask = abs(init_states[i, 0]) > 0
            label = np.array(STATE_NAMES)[mask]
            plot_classifier(init_states[i, :, mask],
                            bool_state[i],
                            x_label=label[0],
                            y_label=label[1],
                            ax=ax
                            )
            plt.tight_layout()

            if not args.one_figure:
                file_path = f'{save_path}/stability_{label[0]}-{label[1]}_th-{th_str}_t-{t}.png'.replace(
                    '$', '').replace('\\', '')
                fig.savefig(file_path)
                print(f'  ==> file {file_path} saved.')

        if args.one_figure:
            file_path = f'{save_path}/stability_th-{th_str}_t-{t}_ord-{args.ord}.png'
            fig.savefig(file_path)
            print(f'  ==> file {file_path} saved.')
