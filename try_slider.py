from utils import plot_classifier
from params import state_space
from params import STATE_NAMES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mpl_interactions.ipyplot as plt


path = 'results_ilqr/T_325/'
states = np.load(path + 'stability_region.npz')['states']
# Ordenar indices
aux = list(map(lambda i: (np.nonzero(states[i, 0, 0])[
               0][0], i), list(range(6))))
aux.sort(key=lambda x: x[0])
indices = [x for _, x in aux]
states = states[indices]

breakpoint()

labels = [('$u$', '$x$'), ('$v$', '$y$'), ('$w$', '$z$'),
          ('$p$', '$\phi$'), ('$q$', '$\\theta$'),
          ('$r$', '$\psi$')
          ]
bool_state = np.apply_along_axis(
    lambda x: np.linalg.norm(x) < 4e-1, -1, states[:, :, -1])
# cluster = np.apply_along_axis(get_color, -1, bool_state)
fig, axes = plt.subplots(
    figsize=(8, 6), nrows=state_space.shape[1]//3, ncols=3, dpi=300,
    sharey=False)
axs = axes.flatten()
mask1 = np.apply_along_axis(lambda x, y: np.greater(
    abs(x), y), -1, states[:, 0, 0], 0)
mask2 = state_space[1] > 0
init_states = states[:, :, 0]
sc = list()
for i in range(init_states.shape[0]):
    mask = abs(init_states[i, 0]) > 0
    label = np.array(STATE_NAMES)[mask]
    aux = plot_classifier(init_states[i, :, mask],
                          bool_state[i], x_label=label[0],
                          y_label=label[1], ax=axs[i],
                          )[1]
    sc.append(aux)


def onChange(value):
    bool_state = np.apply_along_axis(
        lambda x: np.linalg.norm(x) < value, -1, states[:, :, -1])
    for i in range(states.shape[0]):
        sc[i].set_array(bool_state[i])
    fig.canvas.draw_idle()


hslideraxis = fig.add_axes([0.25, 0.1, 0.6, 0.03])
hslider = Slider(hslideraxis,
                 label='$\epsilon$',
                 valmin=np.round(np.linalg.norm(.03 * np.ones(12)), 2),
                 valmax=np.round(np.linalg.norm(np.ones(12)), 2)
                 )  # ,
# valstep=1e-1,
# valinit=4e-1)

hslider.on_changed(onChange)
fig.suptitle('Control iLQR')
plt.show()
