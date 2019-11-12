import numpy as np
from deep_learning_module.DQN import QNetwork
import torch

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3
action_map = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}


# Procedure Value_Iteration(S,A,P,R,θ)
# 2:           Inputs
# 3:                     S is the set of all states
# 4:                     A is the set of all actions
# 5:                     P is state transition function specifying P(s'|s,a)
# 6:                     R is a reward function R(s,a,s')
# 7:                     θ a threshold, θ>0
# 8:           Output
# 9:                     π[S] approximately optimal policy
# 10:                    V[S] value function
# 11:           Local
# 12:                     real array Vk[S] is a sequence of value functions
# 13:                     action array π[S]
# 14:           assign V0[S] arbitrarily
# 15:           k ←0
# 16:           repeat
# 17:                     k ←k+1
# 18:                     for each state s do
# 19:                               Vk[s] = maxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk-1[s'])
# 20:           until ∀s |Vk[s]-Vk-1[s]| < θ
# 21:           for each state s do
# 22:                     π[s] = argmaxa ∑s' P(s'|s,a) (R(s,a,s')+ γVk[s'])
# 23:           return π,Vk
def value_iteration(environment, gamma = 0.99):
    state_space_size = environment.observation_space.n
    action_space_size = environment.action_space.n
    V_k = np.zeros(state_space_size)
    # V_k[state_space_size - 1] = 1.0

    V_k_previous = V_k
    P = environment.P
    policy = np.zeros(state_space_size)

    for _ in range(100):

        for state in range(state_space_size):
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


def create_policy_descriptions(policy):
    policy_descriptions = []
    for i in range(16):
        policy_descriptions.append(action_map[policy[i]])
    return policy_descriptions


def policies_are_equal(policy_1, policy_2):
    for i in range(len(policy_1)):
        if policy_1[i] != policy_2[i]:
            return False
    return True


def policy_iteration(environment, gamma=0.99, max_iterations=2000):
    state_space_size = environment.observation_space.n
    policy = np.zeros(state_space_size)

    for _ in range(max_iterations):
        V = calculate_values_from_policy(policy, environment, gamma)
        new_policy = extract_policy(V, environment, gamma)
        has_not_changed = policies_are_equal(policy, new_policy)
        if has_not_changed:
            break
        policy = new_policy
    return V, policy


def extract_policy(V, env, gamma=0.99):
    actions = [0, 1, 2, 3]
    P = env.P
    s_n = env.observation_space.n
    policy = np.zeros(s_n)
    for i in range(s_n):
        state_transition_list = P[i]
        max = 0.0
        for action in actions:
            sum = 0.0
            action_transition_list = state_transition_list[action]
            for transition in action_transition_list:
                probability = transition[0]
                next_state = transition[1]
                reward = transition[2]
                sum += probability * (reward + gamma * V[next_state])
            if sum > max:
                max = sum
                policy[i] = action
    return policy


def calculate_values_from_policy(policy, env, gamma=0.99, iterations=50):
    #         v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
    #     if (np.sum((np.fabs(prev_v - v))) <= eps):
    P = env.P
    s_n = env.observation_space.n
    V = np.zeros(s_n)
    V[env.observation_space.n - 1] = 1.0
    V_prev = V
    for _ in range(iterations):
        for i in range(s_n):
            state_transition_list = P[i]
            action = policy[i]
            sum = 0.0
            action_transition_list = state_transition_list[action]
            for transition in action_transition_list:
                probability = transition[0]
                next_state = transition[1]
                reward = transition[2]
                sum += probability * (reward + gamma * V_prev[next_state])
            V[i] = sum

        V_prev = V
    return V


def evaluate_policy(env, policy):
    hit = 0
    iterations = 1000
    for _ in range(iterations):
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            if reward > 0.99:
                hit += 1
    return float(hit/iterations)


def get_values_and_policy_from_q_learner(path):
    model = QNetwork(16, 4, 0)
    model.load_state_dict(torch.load(path))
    model.eval()

    policy = np.zeros(16)
    V = np.zeros(16)

    for i in range(16):
        state = np.zeros(16)
        state[i] = 1
        state = torch.from_numpy(state).float().unsqueeze(0).to("cpu")

        action_values = model(state)

        V[i] = np.max(action_values.cpu().data.numpy())

        policy[i] = np.argmax(action_values.cpu().data.numpy())

    return V, policy


def get_values_from_q_learner(path):
    q_table = np.load(path)
    V = np.zeros(16)
    policy = np.zeros(16)

    for i in range(16):
        V[i] = np.max(q_table[i])
        policy[i] = np.argmax(q_table[i])

    return V, policy


def print_array(a):
    print(a[0:4])
    print(a[4:8])
    print(a[8:12])
    print(a[12:16])
