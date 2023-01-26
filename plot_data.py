import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stats():
    # Constant
    my_data = pd.read_csv("record.csv", sep=',', header=0)
    print(my_data.shape)
    x = np.arange(0, my_data.shape[0] * 0.1, 0.1)

    # Prepare the plots
    fig, axes = plt.subplots(nrows=1, ncols=3)
    ax = axes.ravel()
    ax[0].set_title("X-Axis")
    ax[1].set_title("Y-Axis")
    ax[2].set_title("Orientation")

    # Plot function
    def plot(x, y1, y2, index):
        ax[index].plot(x, y1, 'b+', markersize=2, label="Forward Euler")
        ax[index].plot(x, y2, 'r--', markersize=1, label="Midpoint")
        ax[index].legend()
        ax[index].grid()

    plot(x, my_data.iloc[:, 0], my_data.iloc[:, 3], 0)
    plot(x, my_data.iloc[:, 1], my_data.iloc[:, 4], 1)
    plot(x, my_data.iloc[:, 2], my_data.iloc[:, 5], 2)

    plt.show()


if __name__ == '__main__':
    stats()
