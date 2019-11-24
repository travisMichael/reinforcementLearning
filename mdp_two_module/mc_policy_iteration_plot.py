import numpy as np
import matplotlib.pyplot as plt
from mdp_two_module.mountainCarPolicyIteration import policy_iteration


def generate_pi_stats_2():
    gridSize = [100, 100]
    # %gridSize = [200 200];
    gridPos = gridSize[1]
    uMax = 1
    x0 = [-0.52, 0]
    J, policy, scores, times = policy_iteration(uMax, gridSize, 1000, gamma=0.99)
    np.save('stats/policy_iteration_scores_0-99', np.array(scores))
    np.save('stats/policy_iteration_times_0-99', np.array(times))


def policy_iteration_plot_2():

    scores = np.load('stats/policy_iteration_scores_0-99.npy')
    times = np.load('stats/policy_iteration_times_0-99.npy')

    i_list = []
    for i in range(len(scores)):
        i_list.append(i)

    labels = ['Time(s)', 'Iterations']
    values = [times, i_list]
    # values_2 = [times_2, i_list_2]
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=1.0)
    for i in range(1, 3):
        ax = fig.add_subplot(2, 1, i)
        plt.xlabel(labels[i-1])
        plt.ylabel('Avg Score')
        if i == 1:
            plt.title('Policy Iteration')
        ax.plot(values[i - 1], scores)
        # ax.plot(values_2[i-1], scores_2)

    plt.savefig('policy_iteration_plot')
    print()


if __name__ == "__main__":
    # generate_pi_stats_2()
    policy_iteration_plot_2()
