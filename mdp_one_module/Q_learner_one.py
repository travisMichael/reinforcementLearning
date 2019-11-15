import numpy as np
import gym
import sys
from discrete.QLearningAgent import QLearningAgent
from mdp_one_module.util import calculate_error

env = gym.make('FrozenLake-v0')
env.seed(505)


def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    episode_list = []
    avg_score_list = []
    max_avg_score = -np.inf
    avg_score = 0.0
    old_q_table = np.copy(agent.q_table)
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # Save final score
        scores.append(total_reward)
        avg_score_list.append(avg_score)
        episode_list.append(i_episode)
        # Print episode stats
        if mode == 'train':

            if len(scores) > 500:
                avg_score = np.mean(scores[-500:])
                if avg_score > max_avg_score:
                    np.save('best_q_learner_one', agent.q_table)
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                e = calculate_error(old_q_table, agent.q_table)
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, e), end="")
                sys.stdout.flush()
                old_q_table = np.copy(agent.q_table)

    return scores, avg_score_list, episode_list


q_agent = QLearningAgent(env)
scores, avg_score_list, episode_list = run(q_agent, env)

np.save('q_learner_stats/avg_score_list', np.array(avg_score_list))
np.save('q_learner_stats/episode_list', np.array(episode_list))

print()