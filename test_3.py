import numpy as np
import gym
import sys
from util_2 import discretize, create_uniform_grid
from mdp_two_module.snapToGrid import snap_to_grid

env = gym.make('MountainCar-v0')
# env = PendulumEnv()
# env.seed(505)


def run(q_table, env, num_episodes=20000, mode='train'):
    best = -1111111
    P_MIN = -1.2
    P_MAX = 0.5
    V_MIN = -0.07
    V_MAX = 0.07
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        total_reward = 0
        done = False

        # Roll out steps until done
        for _ in range(1000):
            env.render()
            pNext, vNext = state
            pNextIdx = snap_to_grid(pNext, P_MIN, P_MAX, 100)
            vNextIdx = snap_to_grid(vNext, V_MIN, V_MAX, 100)
            action = q_table[int(pNextIdx)][int(vNextIdx)] * (-1) + 1
            state, reward, done, info = env.step(int(action))
            total_reward += reward

    return scores

# state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(100, 100))

q_table = np.load('mdp_two_module/policy_2.npy')
scores = run(q_table, env, mode='test')