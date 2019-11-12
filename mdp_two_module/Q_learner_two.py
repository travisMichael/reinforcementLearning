import numpy as np
import sys
from discrete.DiscreteQLearningAgent import DiscreteQLearningAgent
from mdp_two_module.util_2 import create_uniform_grid
from Pendulum import PendulumEnv


# env = gym.make('Pendulum-v0')
env = PendulumEnv()
env.seed(505)


def run(agent, env, num_episodes=20000, mode='train'):
    best = -1111111
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    rewards = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        for i in range(300):
            state, reward, done, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            action = agent.act(state, reward, done, mode)
            if len(rewards) > 100 and i % 1000 == 0:
                avg_reward = np.mean(rewards[-100:])
                #print(avg_reward)

        # Save final score
        scores.append(total_reward)
        if total_reward > best:
            agent.save()
            best = total_reward
            print(total_reward)

        # Print episode stats
        if mode == 'train':
            avg_score = 0.0
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    agent.save()
                    best = total_reward
                    # print("better")
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, avg_score), end="")
                sys.stdout.flush()

    return scores

action_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=(7,))

state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 10, 10))
q_agent = DiscreteQLearningAgent(env, state_grid, action_grid)
scores = run(q_agent, env)