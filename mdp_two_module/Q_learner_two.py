import numpy as np
import sys
import gym
from discrete.DiscreteQLearningAgent import QLearningAgent
from mdp_two_module.util_2 import create_uniform_grid
from environment.MountainCar import MountainCarEnv

env = gym.make('MountainCar-v0')
env = MountainCarEnv()
env.seed(505)


def run(agent, env, num_episodes=100000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    avg_scores = []
    max_avg_score = np.inf
    avg_score = 800
    iterations_without_improvement = 0
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        for i in range(800):
            state, reward, done, info = env.step(action)
            # total_reward += reward
            total_reward += 1
            action = agent.act(state, reward, done, mode)
            if done:
                break

        # Save final score
        scores.append(total_reward)
        avg_scores.append(avg_score)

        iterations_without_improvement += 1

        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score < max_avg_score:
                    max_avg_score = avg_score
                    iterations_without_improvement = 0
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}, Average Score: {}".format(i_episode, num_episodes, max_avg_score, avg_score), end="")
                sys.stdout.flush()

        if iterations_without_improvement > 5000:
            break

    return avg_scores


def generate_q_learner_2_stats():
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(20, 20))

    # 1
    q_agent = QLearningAgent(env, state_grid, strategy=0)
    scores = run(q_agent, env)
    np.save('q_learner_stats/exp_10000', np.array(scores))
    print('5')

    # 2
    q_agent = QLearningAgent(env, state_grid, strategy=1)
    scores = run(q_agent, env)
    np.save('q_learner_stats/exp_10000', np.array(scores))
    print('5')

    # 3
    q_agent = QLearningAgent(env, state_grid, strategy=2)
    scores = run(q_agent, env)
    np.save('q_learner_stats/exp_10000', np.array(scores))
    print('5')


# q_agent = QLearningAgent(env, state_grid)
# scores = run(q_agent, env)
# np.save('q_learner_stats/alpha_0-005', np.array(scores))
# print('005')

# q_agent = QLearningAgent(env, state_grid, alpha=0.05)
# scores = run(q_agent, env)
# np.save('q_learner_stats/alpha_0-05', np.array(scores))
# print('05')
#
# q_agent = QLearningAgent(env, state_grid, alpha=0.5)
# scores = run(q_agent, env)
# np.save('q_learner_stats/alpha_0-5', np.array(scores))
# print('5')

# q_agent = QLearningAgent(env, state_grid, alpha=0.05, gamma=)
# scores = run(q_agent, env)
# np.save('q_learner_stats/gamma_0-999', np.array(scores))
# print('999')

# q_agent = QLearningAgent(env, state_grid, alpha=0.05, gamma=0.99)
# scores = run(q_agent, env)
# np.save('q_learner_stats/gamma_0-99', np.array(scores))
# print('99')
#
# q_agent = QLearningAgent(env, state_grid, alpha=0.05, gamma=0.9)
# scores = run(q_agent, env)
# np.save('q_learner_stats/gamma_0-9', np.array(scores))
# print('9')
#
# q_agent = QLearningAgent(env, state_grid, alpha=0.05, epsilon_decay_rate=0.999)
# scores = run(q_agent, env)
# np.save('q_learner_stats/geo_0-999', np.array(scores))
# print('geo_99999')
#
# q_agent = QLearningAgent(env, state_grid, alpha=0.05, epsilon_decay_rate=0.99)
# scores = run(q_agent, env)
# np.save('q_learner_stats/geo_0-99', np.array(scores))
# print('geo_99')

# q_agent = QLearningAgent(env, state_grid, alpha=0.05, strategy=1)
# scores = run(q_agent, env)
# np.save('q_learner_stats/linear_0-0001', np.array(scores))
# print('5')

