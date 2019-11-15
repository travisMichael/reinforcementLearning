import gym
from environment.FrozenLake import FrozenLakeEnv
from mdp_one_module.util import *
import matplotlib.pyplot as plt

# env = gym.make('FrozenLake-v0')
# state = env.reset()

env = FrozenLakeEnv()

# print(state)
print(env.action_space)
print(env.observation_space)
s_n = env.observation_space.n
a_n = env.action_space.n


# V_k, policy, e, t = value_iteration(env, 0.99)
# V_k_2, policy_2, e_2, t_2 = value_iteration(env, 0.1)
V_k_2, policy_2, e_2, t_2 = policy_iteration(env, 0.99)
# V_k_3, policy_3 = get_values_and_policy_from_q_learner('checkpoint.pth')
# V_k_3, policy_3 = get_values_from_q_learner('best_q_learner_one.npy')
i_list = []
# e_list = []
for i in range(len(e_2)):
    i_list.append(i)


labels = ['time(s)', 'Iterations']
values = [t_2, i_list]
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=1.0)
for i in range(1, 3):
    ax = fig.add_subplot(2, 1, i)
    plt.xlabel(labels[i-1])
    ax.plot(values[i-1], e_2)

plt.show()

# plt.figure()

# plt.plot(i_list, e_2)
# plt.plot(t_2, e_2)
# plt.xticks(np.arange(min(i_list), max(i_list)+1, 1.0))

# plt.show()

policy_descriptions = create_policy_descriptions(policy)
# policy_descriptions_3 = create_policy_descriptions(policy_3)

evaluation_score = evaluate_policy(env, policy)


# env.render()
print_array(V_k)

print_array(policy_descriptions)
print('------------------------------------')
# print_array(policy_descriptions_3)
print('------')
print(policy)
print(policy_2)
print('------------------------------------')
print(V_k)
print(V_k_2)

env.close()

print()


