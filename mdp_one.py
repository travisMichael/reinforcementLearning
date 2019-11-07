import gym
from FrozenLake import FrozenLakeEnv
from util import *

env = gym.make('FrozenLake-v0')
state = env.reset()

env = FrozenLakeEnv()

print(state)
print(env.action_space)
print(env.observation_space)
s_n = env.observation_space.n
a_n = env.action_space.n


V_k, policy = value_iteration(env, 0.99)
V_k_2, policy_2 = policy_iteration(env, 0.99)
V_k_3, policy_3 = get_values_and_policy_from_q_learner('checkpoint.pth')

policy_descriptions = create_policy_descriptions(policy)
policy_descriptions_3 = create_policy_descriptions(policy_3)

evaluation_score = evaluate_policy(env, policy)


# env.render()
print_array(V_k)

print_array(policy_descriptions)
print('------------------------------------')
print_array(policy_descriptions_3)
print('------')
print(policy)
print(policy_2)
print('------------------------------------')
print(V_k)
print(V_k_2)

env.close()

print()


