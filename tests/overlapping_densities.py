import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--column', type=str, default='distance')
    args = parser.parse_args()

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Create the data
    # rs = np.random.RandomState(1979)
    # x = rs.randn(600)
    # g = np.tile(list("012345"), 100)
    # df = pd.DataFrame(dict(x=x, g=g))
    # m = df.g.map(ord)
    # df["x"] += m
    data = np.load('results2.npz')[args.column]
    # data = np.apply_along_axis(np.linalg.norm, -1, states)

    # Flatten the array
    flattened_arr = data.flatten()
    indices = np.repeat(np.arange(data.shape[0]), data.shape[1])

    # Create DataFrame
    df = pd.DataFrame({args.column: flattened_arr, 'iter': indices})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(data.shape[0], rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="iter", hue="iter",
                      aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, args.column,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, args.column, clip_on=False,
          color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates

    def label(x, color, label):
        ax = plt.gca()
        ax.text(-.1, .3, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
        if args.column == 'distance':
            ax.set_xlim(-200, 1000)
        elif args.column == 'costs':
            ax.set_xlim(-2e6, 1e5)

    g.map(label, args.column)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.show()
