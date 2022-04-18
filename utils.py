import numpy as np
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_state(dic_state, alpha=1, init_state=None, t=None, cmap=None,
               axs=None):
    n = len(axs)
    if isinstance(t, np.ndarray):
        m = next(iter(dic_state.values())).shape
        t = np.linspace(0, 1, m)
    if axs == None:
        _, axs = plt.subplots(np.ceil(n/2), 2, sharex=True, dpi=250)
    if cmap == None:
        cmap = cm.rainbow(np.linspace(0, 1, np.ceil(n/2)))

    list_state = dic_state.items()
    for e, color in enumerate(cmap):
        kx, x = list_state[2 * e]
        kdx, dx = list_state[2 * e + 1]
        axs[e, 0].plot(t, x, c=color, alpha=alpha)
        axs[e, 0].set_ylabel(kx)
        axs[e, 0].set_xlabel('$t$')
        axs[e, 1].plot(t, dx, c=color, alpha=alpha)
        axs[e, 1].set_ylabel(kdx)
        axs[e, 1].set_xlabel('$t$')

        if isinstance(init_state, np.ndarray):
            axs[e, 0].hlines(y=init_state[2 * e], xmin=t[0],
                             xmax=t[-1], linewidth=1, color='b',
                             linestyles='--')
            axs[e, 1].hlines(y=init_state[2 * e + 1], xmin=t[0],
                             xmax=t[-1], linewidth=1, color='b',
                             linestyles='--')

    return axs
