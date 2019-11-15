from environment.FrozenLake import FrozenLakeEnv
from mdp_one_module.util import *
import matplotlib.pyplot as plt
import numpy as np


def q_learner_plot():
    env = FrozenLakeEnv()

    score_list = np.load('q_learner_stats/avg_score_list.npy')
    episode_list = np.load('q_learner_stats/episode_list.npy')

    plt.figure()

    plt.plot(episode_list, score_list)

    plt.savefig('q_learner_plot')
    print()


q_learner_plot()
