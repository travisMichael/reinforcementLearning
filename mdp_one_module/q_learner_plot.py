from environment.FrozenLake import FrozenLakeEnv
from mdp_one_module.util import *
import matplotlib.pyplot as plt
import numpy as np


def q_learner_gamma_plot():

    score_list, episode_list = load('q_learner_stats/avg_score_list.npy')
    score_list_2, episode_list_2 = load('q_learner_stats/avg_score_list_gamma_0-8.npy')
    score_list_3, episode_list_3 = load('q_learner_stats/avg_score_list_gamma_0-6.npy')

    plt.figure()
    plt.title('Q-learning with different\n discount rates')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.plot(episode_list, score_list, color='r', label=r'$\gamma = 0.99$')
    plt.plot(episode_list_2, score_list_2, color='g', label=r'$\gamma = 0.8$')
    plt.plot(episode_list_3, score_list_3, color='b', label=r'$\gamma = 0.6$')

    plt.legend()
    plt.savefig('q_learner_gamma_plot')
    print()


def q_learner_alpha_plot():

    score_list, episode_list = load('q_learner_stats/avg_score_list_alpha_0-9.npy')
    score_list_2, episode_list_2 = load('q_learner_stats/avg_score_list_alpha_0-5.npy')
    score_list_3, episode_list_3 = load('q_learner_stats/avg_score_list_alpha_0-05.npy')
    score_list_4, episode_list_4 = load('q_learner_stats/avg_score_list_alpha_0-005.npy')

    plt.figure()
    plt.title('Q-learning with different\n learning rates')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.plot(episode_list, score_list, color='y', label=r'$\alpha = 0.9$')
    plt.plot(episode_list_2, score_list_2, color='r', label=r'$\alpha = 0.5$')
    plt.plot(episode_list_3, score_list_3, color='g', label=r'$\alpha = 0.05$')
    plt.plot(episode_list_4, score_list_4, color='b', label=r'$\alpha = 0.005$')

    plt.legend()
    plt.savefig('q_learner_alpha_plot')
    print()


def q_learner_exploration_plot():

    score_list, episode_list = load('q_learner_stats/avg_score_list_alpha_0-9.npy')
    score_list_2, episode_list_2 = load('q_learner_stats/avg_score_list_alpha_0-5.npy')
    score_list_3, episode_list_3 = load('q_learner_stats/avg_score_list_alpha_0-05.npy')
    score_list_4, episode_list_4 = load('q_learner_stats/avg_score_list_alpha_0-005.npy')

    plt.figure()
    plt.title('Q-learning with different\n exploration strategies')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.plot(episode_list, score_list, color='y', label=r'$\alpha = 0.9$')
    plt.plot(episode_list_2, score_list_2, color='r', label=r'$\alpha = 0.5$')
    plt.plot(episode_list_3, score_list_3, color='g', label=r'$\alpha = 0.05$')
    plt.plot(episode_list_4, score_list_4, color='b', label=r'$\alpha = 0.005$')

    plt.legend()
    plt.savefig('q_learner_exploration_plot')
    print()


def q_learner_linear_exploration_plot():

    score_list, episode_list = load('q_learner_stats/avg_score_list_linear_0-1.npy')
    score_list_2, episode_list_2 = load('q_learner_stats/avg_score_list_linear_0-01.npy')

    plt.figure()
    plt.title('Q-learning with different exploration strategies with linear decay')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.plot(episode_list, score_list, color='g', label=r'$\mu = 0.1$')
    plt.plot(episode_list_2, score_list_2, color='b', label=r'$\mu = 0.01$')

    plt.legend()
    plt.savefig('q_learner_linear_exploration_plot')
    print()


def q_learner_geometric_exploration_plot():

    score_list_3, episode_list_3 = load('q_learner_stats/avg_score_list_geometric_0-995.npy')
    score_list_4, episode_list_4 = load('q_learner_stats/avg_score_list_geometric_0-85.npy')
    score_list_5, episode_list_5 = load('q_learner_stats/avg_score_list_geometric_0-99.npy')

    plt.figure()
    plt.title('Q-learning with different exploration strategies\n with geometric decay')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.plot(episode_list_3, score_list_3, color='r', label=r'$\mu =  0.995$')
    plt.plot(episode_list_4, score_list_4, color='g', label=r'$\mu = 0.9999$')
    plt.plot(episode_list_5, score_list_5, color='b', label=r'$\mu = 0.99$')

    plt.legend()
    plt.savefig('q_learner_geometric_exploration_plot')
    print()


def q_learner_exp_exploration_plot():

    score_list_6, episode_list_6 = load('q_learner_stats/avg_score_list_exp_2200.npy')
    score_list_7, episode_list_7 = load('q_learner_stats/avg_score_list_exp_10000.npy')
    score_list_8, episode_list_8 = load('q_learner_stats/avg_score_list_exp_500.npy')

    plt.figure()
    plt.title('Q-learning with different exploration strategies\n with exponential decay')
    plt.xlabel('Episode')
    plt.ylabel('Avg Score')

    plt.plot(episode_list_8, score_list_8, color='r', label=r'$\mu = 500$')
    plt.plot(episode_list_6, score_list_6, color='g', label=r'$\mu = 2200$')
    plt.plot(episode_list_7, score_list_7, color='b', label=r'$\mu = 10000$')

    plt.legend()
    plt.savefig('q_learner_exp_exploration_plot')
    print()


def load(path):
    scores = np.load(path)

    episodes = []
    for i in range(len(scores)):
        episodes.append(i)
    return scores, episodes


# q_learner_gamma_plot()
# q_learner_alpha_plot()
q_learner_linear_exploration_plot()
q_learner_geometric_exploration_plot()
q_learner_exp_exploration_plot()
