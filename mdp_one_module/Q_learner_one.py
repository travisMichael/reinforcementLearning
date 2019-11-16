import numpy as np
import gym
import sys
from discrete.QLearningAgent import QLearningAgent
from mdp_one_module.util import calculate_error

env = gym.make('FrozenLake-v0')
env.seed(505)


def run(agent, env, num_episodes=100000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    avg_score_list = []
    max_avg_score = -np.inf
    avg_score = 0.0
    iterations_without_improvement = 0
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        iterations_without_improvement += 1
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
        # Print episode stats
        if mode == 'train':

            if len(scores) > 500:
                avg_score = np.mean(scores[-500:])
                if avg_score > max_avg_score:
                    np.save('best_q_learner_one', agent.q_table)
                    max_avg_score = avg_score
                    iterations_without_improvement = 0
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()
                old_q_table = np.copy(agent.q_table)

        if iterations_without_improvement > 5000:
            break

    return scores, avg_score_list


q_agent = QLearningAgent(env, strategy=2)
scores, avg_score_list = run(q_agent, env)

np.save('q_learner_stats/avg_score_list_exp_10000', np.array(avg_score_list))


print()