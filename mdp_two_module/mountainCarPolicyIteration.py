import numpy as np
from mdp_two_module.extractPolicy import extract_policy_from_values
from mdp_two_module.extractValuesFromPolicy import extract_values_from_policy
from mdp_two_module.score_policy import score_policy
from time import time


def policies_are_equal(policy_1, policy_2):

    count = 0
    for i in range(len(policy_1[0])):
        for j in range(len(policy_1[1])):
            if policy_1[i][j] != policy_2[i][j]:
                count += 1

    return count


def policy_iteration(uMax, gridSize, maxHorizon, gamma=0.99):
    gridPos = gridSize[0]
    gridVel = gridSize[1]

    score_list = []
    time_list = []
    time_sum = 0

    policy = np.zeros((gridPos+1, gridVel+1))

    for index in range(100):
        start = time()
        J, policy = extract_values_from_policy(uMax, gridSize, maxHorizon,  policy, gamma)
        new_policy = extract_policy_from_values(uMax, gridSize, J)

        c = policies_are_equal(policy, new_policy)
        time_sum += time() - start
        if c > 0:
            print(c)
        else:
            break
        policy = np.copy(new_policy)
        if index % 2 == 0:
            score_list.append(score_policy(policy, 100, 100))
            time_list.append(time_sum)

    return J, policy, score_list, time_list


# gridSize = [100, 100]
# # %gridSize = [200 200];
# gridPos = gridSize[1]
# uMax = 1
# x0 = [-0.52, 0]
#
#
#
# J, policy = policy_iteration(uMax, gridSize, 1000)
#
# np.save('policy_2', policy)