import sys
from mdp_one_module.value_iteration_plot import value_iteration_plot
from mdp_one_module.policy_iteration_plot import policy_iteration_plot
from mdp_one_module.Q_learner_one import generate_q_learner_stats
#
from mdp_two_module.value_iteration_plot import value_iteration_plot_2, generate_vi_stats_2
from mdp_two_module.mc_policy_iteration_plot import policy_iteration_plot_2, generate_pi_stats_2
from mdp_two_module.Q_learner_two import generate_q_learner_2_stats
from mdp_two_module.q_learner_plot import *


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please specify filename and data set to pre-process")
    else:
        plot = sys.argv[1]
        if plot == 'vi_mdp_one':
            value_iteration_plot()
        elif plot == 'pi_mdp_one':
            policy_iteration_plot()
        elif plot == 'q_learner_one':
            generate_q_learner_stats()
        elif plot == 'vi_mdp_two':
            generate_vi_stats_2()
            value_iteration_plot_2()
        elif plot == 'pi_mdp_two':
            policy_iteration_plot_2()
            generate_pi_stats_2()
        elif plot == 'q_learner_two':
            generate_q_learner_2_stats()
            q_learner_2_gamma_plot()
            q_learner_2_alpha_plot()
            q_learner_2_exponential_exploration_plot()
