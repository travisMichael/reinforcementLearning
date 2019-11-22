import numpy as np


def discretize(sample, grid):
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def create_uniform_grid(low, high, bins=(10, 10, 10)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid
