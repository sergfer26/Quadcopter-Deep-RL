# import os
import pytz
import seaborn as sns
# import pathlib
import numpy as np
# from GPS.utils import Memory
import pandas as pd
from mycolorpy import colorlist as mcp
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from datetime import datetime
# from simulation import n_rollouts


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


"""
def load_memory(expert, env, n, path='Linear/', n_x=None, n_u=None, t_x=None, t_u=None):
    '''
Carga el almacenamiento de las demostraciones del experto.

Argumenetos
-----------
env: QuadcopterEnv
Entorno del quadricoptero.
n: int
Número de simulaciones
path: str
Dirección donde será/esta guardada la memoria.
n_x, n_u: int
Dimensión de los estados y las acciones respectivamente.
t_x, t_u: function
Transformaciones aplicadas a las salidas de `env`.

Retornos
--------
memory: `GCL.utils.Memory``
Es el almacenamiento de las demostraciones del experto.
'''
    if n_x is None:
        n_x = env.observation_space.shape[0]
    if n_u is None:
        n_u = env.action_space.shape[0]
    memory = Memory(max_size=n, action_dim=n_u, state_dim=n_x, T=env.steps - 1)
    if os.path.exists(path + 'memory_transformed_x_u.npz'):
        arrays = np.load(path + 'memory_transformed_x_u.npz')
        states = arrays['states']
        actions = arrays['actions']
        new_states = arrays['next_states']
        dones = arrays['dones']
    else:
        # expert = LinearAgent(env)
        states, actions, _ = n_rollouts(expert, env, n)
        if callable(t_x):
            states = np.apply_along_axis(t_x, -1, states)
        if callable(t_u):
            actions = np.apply_along_axis(t_u, -1, actions)
        new_states = states[:, 1:, :]
        states = states[:, :-1, :]
        dones = np.zeros((n, env.steps - 1), dtype=bool)
        dones[:, -1] = True
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        kwargs = dict(states=states, actions=actions,
                      next_states=new_states, dones=dones)
        np.savez(path + 'memory_transformed_x_u.npz', **kwargs)
    memory.push(states=states, actions=actions,
                next_states=new_states, dones=dones)
    return memory
"""


def date_as_path():
    tz = pytz.timezone('America/Mexico_City')
    mexico_now = datetime.now(tz)
    year = str(mexico_now.year)[-2:]
    month = str(mexico_now.month)
    day = str(mexico_now.day)
    hour = str(mexico_now.hour)
    minut = str(mexico_now.minute)
    month = month if len(month) == 2 else '0' + month
    day = day if len(day) == 2 else '0' + day
    hour = hour if len(hour) == 2 else '0' + hour
    minut = minut if len(minut) == 2 else '0' + minut
    return year + '_' + month + '_' + day + '_' + hour + '_' + minut


def plot_performance(*args, xlabel=None, ylabel=None,
                     title='', ax=None, figsize=(7, 6), dpi=150,
                     cmap='winter', alphas=None, colors=None, labels=None
                     ):
    '''
    *args : `np.arrary`
        Arreglo(s) a graficar.
    xlabel : `str`
        Etiqueta del eje x.
    ylabel : `str`
        Etiqueta del eje y.
    ax : `plt.Axes`
        Eje de gráfica.
    figsize : `tuple`
        Dimensiones de la figura
    dpi : int
        Número de pixeles.
    cmp : str
        Nombre del colormap de matplotlib:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    alphas : `list`o `np.ndarray`
        Arreglo de las transparencis de las curvas.
    colors : `list`
        Lista de colores para graficar.
    labels : `list`
        Lista de las etiquetas de las curvas.
    '''
    if not isinstance(colors, list):
        colors = mcp.gen_color(cmap=cmap, n=len(args))
    if not isinstance(alphas, list):
        alphas = np.linspace(0.1, 1, len(args))
        if len(alphas) == 1:
            alphas = [1]
    if not isinstance(labels, list):
        labels = [None] * len(args)
    if not isinstance(xlabel, str):
        xlabel = '$x$'
    if not isinstance(ylabel, str):
        ylabel = '$y$'
    fig = None
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for e, x in enumerate(args):
        ax.plot(x, color=colors[e], alpha=alphas[e], label=labels[e])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    return fig, ax


def plot_state(dic_state, alpha=1, init_state=None, t=None, cmap=None,
               axs=None, style="seaborn-whitegrid"):
    plt.style.use(style)
    n = len(dic_state)
    if isinstance(t, np.ndarray):
        m = list(dic_state.values())[0].shape[0]
        t = np.linspace(0, 1, m)

    if axs is None:
        _, axs = plt.subplots(int(np.ceil(n/2)), 2,
                              figsize=(6, 18), sharex=True, dpi=250)
    if cmap is None:
        cmap = cm.rainbow(np.linspace(0, 1, int(np.ceil(n/2))))

    list_state = list(dic_state.items())
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


def violin_plot(x_name='x', y_name='y', hue=None, split=True, ax=None,
                style="seaborn-whitegrid", **kwargs):
    # (K, M)
    plt.style.use(style)
    columns = [x_name, y_name]
    if isinstance(hue, str):
        columns += [hue]
    else:
        split = False
    data = pd.DataFrame(columns=columns)
    for key, array in kwargs.items():
        K = array.shape[1]
        M = array.shape[0]
        labels = list()
        for i in range(K):
            labels += [i for _ in range(M)]
        labels = np.array(labels, dtype=int)
        array = array.flatten()
        array = np.stack([labels, array], axis=1)

        aux = pd.DataFrame(data=array, columns=[x_name, y_name])
        if isinstance(hue, str):
            aux[hue] = key
        data = pd.concat([data, aux])
    return sns.violinplot(data=data, x=x_name, y=y_name, hue=hue,
                          split=split, ax=ax)
