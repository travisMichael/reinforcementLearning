from environment.FrozenLake import FrozenLakeEnv
from mdp_one_module.util import *
import matplotlib.pyplot as plt


def value_iteration_plot(env):

    V_k, policy, e, t = value_iteration(env, 0.99)
    i_list = []
    for i in range(len(e)):
        i_list.append(i)

    labels = ['time(s)', 'Iterations']
    values = [t, i_list]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=1.0)
    for i in range(1, 3):
        ax = fig.add_subplot(2, 1, i)
        plt.xlabel(labels[i-1])
        ax.plot(values[i-1], e)

    plt.savefig('value_iteration_plot.png')

    print()


env = FrozenLakeEnv()
value_iteration_plot(env)
