# from CartPole import CartPoleEnv
from environment.MountainCar import MountainCarEnv
import numpy as np
from mdp_two_module.util_2 import create_uniform_grid
from mdp_two_module.util_2 import discretize

# env = gym.make('MountainCar-v0')
env = MountainCarEnv()
# env = PendulumEnv()
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
    d_state = discretize(state, state_grid)
    # points = get_neighboring_points(d_state, env)
    # next_state_map = {}
    best_value = -np.inf
    best_action = 0
    reward = -1.0
    for action in action_grid:
        env.reset()
        env.state = state
        next_state, reward, done, _ = env.step(action)
        n_s = tuple(discretize(next_state, state_grid))
        value = gamma * V_k_prev[n_s[0]][n_s[1]]
        if done and reward != -10:
            reward = 0.0
            # env.render()

        # value = reward + gamma * V_k_prev[n_s[0]][n_s[1]]
        if value == best_value and np.random.uniform(0, 1) > 0.33:
            best_value = value
            best_action = action
        if value > best_value:
            best_value = value
            best_action = action

    return reward + best_value, best_action


def V_diff(V_1, V_2):
    diff = (np.square(V_1 - V_2)).mean(axis=None)
    return diff


def value_iteration(env, state_grid, action_grid, gamma=0.99):
    state_size = tuple(len(splits) + 1 for splits in state_grid)  # n-dimensional state space

    V_k = np.zeros(shape=(state_size))
    policy = np.zeros(shape=(state_size))
    policy[:,:] = 0
    V_k_prev = np.copy(V_k)
    iterations = 0
    diff = 1.0
    while diff > 0.000000001:
        iterations += 1
        for i in range(10):
            state = env.observation_space.sample()
            s = tuple(discretize(state, state_grid))
            V_k[s[0]][s[1]], policy[s[0]][s[1]] = calculate_state_value(V_k_prev, state, state_grid, env, action_grid, gamma)
        diff = V_diff(V_k, V_k_prev, state_grid)
        V_k_prev = np.copy(V_k)
        if iterations % 20:
            print(diff)
            np.save('mdp_2_policy', policy)
    print(iterations)
    return V_k, policy


action_grid = [0, 1, 2]
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10))

V_k, policy = value_iteration(env, state_grid, action_grid, 0.99)

np.save('mdp_2_policy', policy)

# _, reward, done, _ = env.step(action)
# next_state = env.state
# print(next_state, reward, done)
#
# env.state = state
#
# _, reward, done, _ = env.step(action_2)
# next_state = env.state
# print(next_state, reward, done)
#
# _, reward, done, _ = env.step(action_3)
# next_state = env.state
# print(next_state, reward, done)
#
# _, reward, done, _ = env.step(action_4)
# next_state = env.state
# print(next_state, reward, done)

env.close()
