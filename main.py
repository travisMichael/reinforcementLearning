import sys
from mdp_one_module.value_iteration_plot import value_iteration_plot
from mdp_one_module.policy_iteration_plot import policy_iteration_plot
from mdp_one_module.Q_learner_one import generate_q_learner_stats


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
