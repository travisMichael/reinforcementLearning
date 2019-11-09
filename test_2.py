import numpy as np
import gym
import sys
from DiscreteQLearningAgent import DiscreteQLearningAgent
from util_2 import create_uniform_grid
from util_2 import discretize



env = gym.make('Pendulum-v0')
env.seed(505)
num_episodes = 50

action_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=(9,))

state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10, 10))

policy = np.load('mdp_2_policy.npy')

scores = []
max_avg_score = -np.inf
for i_episode in range(1, num_episodes+1):
    # Initialize episode
    state = env.reset()

    r = tuple(discretize(state, state_grid))
    action = policy[r[0]][r[1]][r[2]]
    total_reward = 0
    done = False

    # Roll out steps until done
    for _ in range(200):
        env.render()
        state, reward, done, info = env.step([action])
        total_reward += reward
        r = tuple(discretize(state, state_grid))
        action = policy[r[0]][r[1]][r[2]]

