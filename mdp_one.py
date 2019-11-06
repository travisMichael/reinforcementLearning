import gym
# from gym import envs
import numpy as np
from FrozenLake import FrozenLakeEnv

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
v_c = np.zeros(16)
v_c[15] = 1.0
v_old = v_c

hit = 0

for i in range(s_n):
    max = -1.0

for _ in range(10):

    env.reset()
    env.reset_state(5)
    env.render()
    new_state, reward, done, info = env.step(0)
    print(new_state, info['prob'], reward, done)
    if new_state == 0:
        hit += 1

print(0, float(hit/100))

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