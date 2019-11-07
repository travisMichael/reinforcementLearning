import gym
# from gym import envs
import numpy as np
from FrozenLake import FrozenLakeEnv
from util import create_policy_descriptions, value_iteration, policy_iteration, evaluate_policy, print_array

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

policy_descriptions = create_policy_descriptions(policy)

evaluation_score = evaluate_policy(env, policy)


# env.render()
print_array(V_k)

print_array(policy_descriptions)

# print(policy_descriptions[0:4])
# print(policy_descriptions[4:8])
# print(policy_descriptions[8:12])
# print(policy_descriptions[12:16])
print('------')
print(policy)
print(policy_2)
print('------------------------------------')
print(V_k)
print(V_k_2)

env.close()

print()


