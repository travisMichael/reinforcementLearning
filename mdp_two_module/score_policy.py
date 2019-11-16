import numpy as np
import gym
from discrete.snapToGrid import snap_to_grid
from environment.MountainCar import MountainCarEnv


def score_policy(policy, P_MIN, V_MIN, P_MAX, V_MAX, gridPos, gridVel, num_episodes=100):
    # env = gym.make('MountainCar-v0')
    env = MountainCarEnv()
    env.seed(505)

    scores = []

    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        total_reward = 0
        done = False

        # Roll out steps until done
        for _ in range(1000):
            # env.render()
            position = snap_to_grid(state[0], P_MIN, P_MAX, gridPos)
            velocity = snap_to_grid(state[1], V_MIN, V_MAX, gridVel)
            action = policy[int(position)][int(velocity)] + 1
            state, reward, done, info = env.step(int(action))
            total_reward += 1
            if done:
                break

        # Save final score
        scores.append(total_reward)

    return np.mean(scores)

