import gym
# from CartPole import CartPoleEnv
from MountainCar import MountainCarEnv
from Pendulum import PendulumEnv
from util_2 import value_iteration
import numpy as np
from util_2 import create_uniform_grid
import math
from util_2 import discretize


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


def calculate_state_value(V_k_prev, state, state_grid, env, action_grid, gamma):
    max = -np.inf
    best_action = action_grid[0][0]
    for action in action_grid[0]:
        env.state = state
        next_state, reward, _, _ = env.step([action])
        n_s = tuple(discretize(next_state, state_grid))
        value = reward + gamma * V_k_prev[n_s[0]][n_s[1]][n_s[2]]
        if value == max:
            if np.random.uniform(0, 1) < 0.5:
                max = value
                best_action = action
        if value > max:
            max = value
            best_action = action
    return max, best_action


def V_diff(V_1, V_2, state_grid):
    diff = (np.square(V_1 - V_2)).mean(axis=None)
    return diff


def value_iteration(env, state_grid, action_grid, gamma=0.99):
    state_size = tuple(len(splits) + 1 for splits in state_grid)  # n-dimensional state space
    action_size = len(action_grid[0]) # self.env.action_space.n

    V_k = np.zeros(shape=(state_size))
    policy = np.zeros(shape=(state_size))
    V_k[:,:,:] = -50
    V_k_prev = np.copy(V_k)
    iterations = 0
    diff = 1.0
    while diff > 0.0000000001:
        iterations += 1
        for i in range(len(state_grid[0])):
            for j in range(len(state_grid[1])):
                for k in range(len(state_grid[2])):
                    theta = math.atan2(state_grid[0][i], state_grid[1][j])
                    thetadot = state_grid[2][k]
                    state = [theta, thetadot]
                    V_k[i][j][k], policy[i][j][k] = calculate_state_value(V_k_prev, state, state_grid, env, action_grid, gamma)
        diff = V_diff(V_k, V_k_prev, state_grid)
        V_k_prev = np.copy(V_k)
        if iterations % 20:
            print(diff)
    print(iterations)
    return V_k, policy


action_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=(7,))
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10, 10))

V_k, policy = value_iteration(env, state_grid, action_grid, 0.99)

np.save('mdp_2_policy', policy)

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
