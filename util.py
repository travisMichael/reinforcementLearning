import numpy as np


def value_iteration(environment, gamma = 0.99):
    state_space_size = environment.observation_space.n
    action_space_size = environment.action_space.n
    V_k = np.zeros(state_space_size)
    V_k[state_space_size - 1] = 1.0

    V_k_previous = V_k
    P = environment.P
    policy = np.zeros(state_space_size)

    for _ in range(100):

        for state in range(15):
            V_k[state], policy[state] = value_max_with_arg_max(state, V_k_previous, gamma, P)

        V_k_previous = V_k

    return V_k, policy


def value_max_with_arg_max(state, V_k_previous, gamma, P):
    actions = [0, 1, 2, 3]

    state_transition_info = P[state]

    max = 0.0
    best_action = 0
    for action in actions:
        action_transition_info_list = state_transition_info[action]
        sum = 0.0
        for j in range(len(action_transition_info_list)):
            action_transition_info = action_transition_info_list[j]
            transition_probability = action_transition_info[0]
            next_state = action_transition_info[1]
            reward = action_transition_info[2]

            sum += transition_probability * (reward + gamma * V_k_previous[next_state])

        if sum > max:
            max = sum
            best_action = action
    return max, best_action


def create_policy_descriptions(action_map, policy):
    policy_descriptions = []
    for i in range(16):
        policy_descriptions.append(action_map[policy[i]])
