import numpy as np
import gym


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
            action = policy[state]
            state, reward, done, info = env.step(int(action))
            total_reward += reward

        # Save final score
        scores.append(total_reward)

    return np.mean(scores)

