import numpy as np
from mdp_two_module.mountainCarSimulation import mountainCarSim
from discrete.snapToGrid import snap_to_grid
from mdp_two_module.score_policy import score_policy
from time import time


def value_iteration(u_max, gridSize, maxHorizon, should_score=False, gamma=0.99):
    #  This function will do the value iteration process to find the optimal
    #  policy.
    score_list = []
    time_list = []
    time_sum = 0.0
    P_MIN = -1.2
    P_MAX = 0.5
    V_MIN = -0.07
    V_MAX = 0.07
    # Very small number
    EPSILON = 1E-6
    gridPos = gridSize[0]
    gridVel = gridSize[1]

    nPosStates = gridPos + 1
    nVelStates = gridVel + 1

    posGridStep = (P_MAX - P_MIN) / gridPos
    velGridStep = (V_MAX - V_MIN) / gridVel

    J = np.zeros((gridPos + 1, gridVel + 1))
    # Policy matrix
    policy = np.zeros((gridPos+1, gridVel+1))
    u = [0, u_max, -u_max]
    print('value iteration is running...\n')

    for index in range(maxHorizon):
        start = time()
        Jprev = np.copy(J)

        for vIdx in range(nVelStates):
            for pIdx in range(nPosStates):

                p = (pIdx) * posGridStep + P_MIN
                v = (vIdx) * velGridStep + V_MIN

                Jplus1 = np.inf

                for uIdx in u:
                    pNext, vNext = mountainCarSim(p, v, u[uIdx])
                    pNextIdx = snap_to_grid(pNext, P_MIN, P_MAX, gridPos)
                    vNextIdx = snap_to_grid(vNext, V_MIN, V_MAX, gridVel)

                    Jplus1_ =  J[pNextIdx, vNextIdx] * gamma

                    if pNextIdx != gridPos:
                        Jplus1_ = Jplus1_ + 1
                    # if pNextIdx == 0 and vNext < 0:
                    #     Jplus1_ = Jplus1_ + 199

                    if Jplus1_ < Jplus1:
                        Jplus1 = Jplus1_
                        uMinIdx = uIdx
                J[pIdx, vIdx] = Jplus1
                policy[pIdx, vIdx] = uMinIdx
        time_sum += time() - start
        if should_score and index % 10 == 0:
            score_list.append(score_policy(policy,P_MIN, V_MIN, P_MAX, V_MAX, gridPos, gridVel))
            time_list.append(time_sum)
        diff = (np.square(J - Jprev)).mean(axis=None)
        print('episode: ', diff)

        if (diff < EPSILON):
            print('converged with error = %f after %i episodes', index)
            break
    return J, policy, score_list, time_list


