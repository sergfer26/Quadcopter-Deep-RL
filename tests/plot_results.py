import numpy as np
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='distance')
    parser.add_argument('--file-array', type=str, default='results2.npz')
    args = parser.parse_args()
    # Generate random data
    data = np.load(args.file_array)[args.name]

    # Generate some sample data
    time_steps = 6
    num_samples = 800

    # Calculate means, standard deviations, and quantiles
    means = np.mean(data, axis=1)
    std_devs = np.std(data, axis=1)
    # Calculate 25th and 75th percentiles
    quantiles = np.percentile(data, q=[25, 75], axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plotting the mean line
    plt.plot(np.arange(time_steps), means, marker='.',
             color='blue', label='Media')

    # Plotting the shaded area representing the standard deviation
    plt.fill_between(np.arange(time_steps), means - std_devs, means +
                     std_devs, color='blue', alpha=0.2, label='Desviación estándar')

    # Plotting the shaded area representing the interquartile range (25th to 75th percentile)
    plt.fill_between(np.arange(
        time_steps), quantiles[0], quantiles[1], color='orange', alpha=0.5, label='Rango intercuartil')

    # Adding labels and title
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    # plt.title(
    #     'Time Series with Shaded Area for Standard Deviation and Interquartile Range (More Highlighted)')
    plt.legend(loc='lower right')

    plt.grid(True)
    plt.show()
