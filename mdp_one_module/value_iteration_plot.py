from environment.FrozenLake import FrozenLakeEnv
from mdp_one_module.util import *
import matplotlib.pyplot as plt


def value_iteration_plot():

    env = FrozenLakeEnv()

    V_k_2, policy_2, scores, times = value_iteration(env, True, 0.99)

    i_list = []
    for i in range(len(scores)):
        i_list.append(i * 5)
    # for i in range(len(times)):
    #     t_list.append(t_list[i] + times[i])

    labels = ['Time(s)', 'Iterations']
    values = [times, i_list]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=1.0)
    for i in range(1, 3):
        ax = fig.add_subplot(2, 1, i)
        plt.xlabel(labels[i-1])
        plt.ylabel('Avg Score')
        if i == 1:
            plt.title('Value Iteration')
        ax.plot(values[i - 1], scores)

    plt.savefig('value_iteration_plot')
    print()


env = FrozenLakeEnv()
value_iteration_plot()
