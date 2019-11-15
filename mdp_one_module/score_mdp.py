import numpy as np
import gym
# import sys
# from discrete.QLearningAgent import QLearningAgent


def score_policy(policy, num_episodes=1000):
    env = gym.make('FrozenLake-v0')
    env.seed(505)

    scores = []

    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            action = np.argmax(policy[state])
            state, reward, done, info = env.step(action)
            total_reward += reward

        # Save final score
        scores.append(total_reward)

    return scores

