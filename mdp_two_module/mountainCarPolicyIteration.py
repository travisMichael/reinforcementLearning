import numpy as np
from mdp_two_module.extractPolicy import extract_policy_from_values
from mdp_two_module.extractValuesFromPolicy import extract_values_from_policy


def policies_are_equal(policy_1, policy_2):

    count = 0
    for i in range(len(policy_1[0])):
        for j in range(len(policy_1[1])):
            if policy_1[i][j] != policy_2[i][j]:
                count += 1

    return count


def policy_iteration(uMax, gridSize, maxHorizon):
    gridPos = gridSize[0]
    gridVel = gridSize[1]

    policy = np.zeros((gridPos+1, gridVel+1))

    for _ in range(50):
        J, policy = extract_values_from_policy(uMax, gridSize, maxHorizon,  policy)
        new_policy = extract_policy_from_values(uMax, gridSize, J)

        c = policies_are_equal(policy, new_policy)
        if c > 0:
            print(c)
        else:
            break
        policy = np.copy(new_policy)

    return J, policy


gridSize = [100, 100]
# %gridSize = [200 200];
gridPos = gridSize[1]
uMax = 1
x0 = [-0.52, 0]



J, policy = policy_iteration(uMax, gridSize, 1000)

np.save('policy_2', policy)