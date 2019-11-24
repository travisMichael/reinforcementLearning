import numpy as np
from mdp_two_module.util_2 import discretize

GEOMETRIC_ACTION_STRATEGY = 0
LINEAR_ACTION_STRATEGY = 1
EXPONENTIAL_ACTION_STRATEGY = 2
UPPER_CONFIDENCE_BOUND_ACTION_STRATEGY = 3


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, strategy=GEOMETRIC_ACTION_STRATEGY, alpha=0.005, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9999, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.strategy = strategy
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.episode = 0
        self.linear_epsilon = epsilon
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon

        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        # TODO: Implement this
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.episode += 1
        self.linear_epsilon = 0.01 # max(0.01, self.linear_epsilon - 0.0001)
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                                                                   (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

            action = self.act_with_exploration(state)

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

    def act_with_exploration(self, state):
        action = -1
        if self.strategy == GEOMETRIC_ACTION_STRATEGY:
            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])
        elif self.strategy == LINEAR_ACTION_STRATEGY:
            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.linear_epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])
        elif self.strategy == EXPONENTIAL_ACTION_STRATEGY:
            epsilon = np.max([0.1, np.exp(-1.0 * self.episode/1000.0)])
            do_exploration = np.random.uniform(0, 1) < epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])
        # elif self.strategy == UPPER_CONFIDENCE_BOUND_ACTION_STRATEGY:
        #     best_action = 0
        #     best_value = -np.inf
        #     best_natural_value = -np.inf
        #     best_natural_action = 0
        #
        #     for i in range(len(self.q_table[state])):
        #         value = self.q_table[state][i]
        #         if value > best_natural_value:
        #             best_natural_value = value
        #             best_natural_action = i
        #         count = self.action_count_table[state][i]
        #         if count > 0:
        #             value += 0.1 * math.sqrt(math.log2(self.t) / self.action_count_table[state][i])
        #
        #         if value > best_value:
        #             best_action = i
        #             best_value = value
        #
        #     if best_action != best_natural_action:
        #         print('exploring')
        #
        #     action = best_action
        else:
            print('The action strategy is not implemented')

        return action
