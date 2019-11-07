import gym
# from gym import envs
import numpy as np
from FrozenLake import FrozenLakeEnv
from util import create_policy_descriptions, value_iteration, policy_iteration, evaluate_policy

env = gym.make('FrozenLake-v0')
state = env.reset()

env = FrozenLakeEnv()

print(state)
print(env.action_space)
print(env.observation_space)
s_n = env.observation_space.n
a_n = env.action_space.n

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
action_map = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}

V_k, policy = value_iteration(env, 0.99)
V_k_2, policy_2 = policy_iteration(env, 0.99)

policy_descriptions = create_policy_descriptions(action_map, policy)

evaluation_score = evaluate_policy(env, policy)


# env.render()
print(V_k[0:4])
print(V_k[4:8])
print(V_k[8:12])
print(V_k[12:16])
print(policy_descriptions[0:4])
print(policy_descriptions[4:8])
print(policy_descriptions[8:12])
print(policy_descriptions[12:16])
print('------')
print(policy)

# print(envs.registry.all())

env.close()

print()




# Procedure Value_Iteration(S,A,P,R,θ)
# 2:           Inputs
# 3:                     S is the set of all states
# 4:                     A is the set of all actions
# 5:                     P is state transition function specifying P(s'|s,a)
# 6:                     R is a reward function R(s,a,s')
# 7:                     θ a threshold, θ>0
# 8:           Output
# 9:                     π[S] approximately optimal policy
# 10:                    V[S] value function
# 11:           Local
# 12:                     real array Vk[S] is a sequence of value functions
# 13:                     action array π[S]
# 14:           assign V0[S] arbitrarily
# 15:           k ←0
# 16:           repeat
# 17:                     k ←k+1
# 18:                     for each state s do
# 19:                               Vk[s] = maxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])
# 20:           until ∀s |Vk[s]-Vk-1[s]| < θ
# 21:           for each state s do
# 22:                     π[s] = argmaxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk[s'])
# 23:           return π,Vk
