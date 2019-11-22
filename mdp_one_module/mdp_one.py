import gym
from environment.FrozenLake import FrozenLakeEnv
from mdp_one_module.util import *
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
env.render()
# state = env.reset()

env = FrozenLakeEnv()

# print(state)
print(env.action_space)
print(env.observation_space)
s_n = env.observation_space.n
a_n = env.action_space.n


V_k, policy, e, t = value_iteration(env, False, 0.99)
# V_k_2, policy_2, e_2, t_2 = value_iteration(env, 0.1)
# V_k_2, policy_2, e_2, t_2 = policy_iteration(env, 0.99)
# V_k_3, policy_3 = get_values_and_policy_from_q_learner('checkpoint.pth')
# V_k_3, policy_3 = get_values_from_q_learner('best_q_learner_one.npy')


policy_descriptions = create_policy_descriptions(policy)
# policy_descriptions_3 = create_policy_descriptions(policy_3)

evaluation_score = evaluate_policy(env, policy)


# env.render()
print_array(V_k)

print_array(policy_descriptions)
print('------------------------------------')
# print_array(policy_descriptions_3)
print('------')
print(policy)
print('------------------------------------')
print(V_k)

env.close()

print()


