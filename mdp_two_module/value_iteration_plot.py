import numpy as np
import matplotlib.pyplot as plt
from mdp_two_module.mountainCarValueIteration import value_iteration


def generate_vi_stats_2():
    gridSize = [100, 100]
    # %gridSize = [200 200];
    gridPos = gridSize[1]
    uMax = 1
    x0 = [-0.52, 0]
    J, policy, scores, times = value_iteration(uMax, gridSize, 1000, should_score=True, gamma=0.5)
    np.save('stats/value_iteration_scores_0-5', np.array(scores))
    np.save('stats/value_iteration_times_0-5', np.array(times))


def value_iteration_plot_2():

    scores = np.load('stats/value_iteration_scores_0-99.npy')
    times = np.load('stats/value_iteration_times_0-99.npy')

    scores_2 = np.load('stats/value_iteration_scores_0-5.npy')
    times_2 = np.load('stats/value_iteration_times_0-5.npy')

    i_list = []
    for i in range(len(scores)):
        i_list.append(i * 10)

    i_list_2 = []
    for i in range(len(scores_2)):
        i_list_2.append(i * 10)

    labels = ['Time(s)', 'Iterations']
    values = [times, i_list]
    values_2 = [times_2, i_list_2]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=1.0)
    for i in range(1, 3):
        ax = fig.add_subplot(2, 1, i)
        plt.xlabel(labels[i-1])
        plt.ylabel('Avg Score')
        if i == 1:
            plt.title('Value Iteration')
        ax.plot(values[i - 1], scores)
        ax.plot(values_2[i-1], scores_2)

    plt.savefig('value_iteration_plot')
    print()


if __name__ == "__main__":
    generate_stats_2()
    value_iteration_plot_2()
