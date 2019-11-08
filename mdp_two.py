import gym
# from CartPole import CartPoleEnv
from MountainCar import MountainCarEnv
from Pendulum import PendulumEnv
from util_2 import value_iteration
import numpy as np
from util_2 import create_uniform_grid

# env = gym.make('MountainCar-v0')
env = MountainCarEnv()
env = PendulumEnv()
env.seed(505)


env.reset()
state = env.state
# state = env.state

print(env.observation_space.low)
print('--')
print(env.observation_space.high)
print('--')
print(env.action_space)
print('--')

action = env.action_space.sample()
action = np.array([0.2])
action_2 = np.array([-0.2])
action_3 = np.array([-0.6])
action_4 = np.array([0.4])

_, reward, done, _ = env.step(action)
next_state = env.state
print(next_state, reward, done)

env.state = state

_, reward, done, _ = env.step(action_2)
next_state = env.state
print(next_state, reward, done)

_, reward, done, _ = env.step(action_3)
next_state = env.state
print(next_state, reward, done)

_, reward, done, _ = env.step(action_4)
next_state = env.state
print(next_state, reward, done)

env.close()
