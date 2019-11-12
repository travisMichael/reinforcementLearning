

import numpy as np
from mdp_two_module.mountainCarSimulation import mountainCarSim
from discrete.snapToGrid import snap_to_grid


def value_iteration(u_max, gridSize, maxHorizon):
    #  This function will do the value iteration process to find the optimal
    #  policy.
    P_MIN = -1.2
    P_MAX = 0.5
    V_MIN = -0.07
    V_MAX = 0.07
    # Very small number
    EPSILON = 1E-6
    gridPos = gridSize[0]
    gridVel = gridSize[1]
    # Grid size of 3 means the grid consists of: 1 - 2 - 3 - 4
    nPosStates = gridPos + 1
    nVelStates = gridVel + 1

    posGridStep = (P_MAX - P_MIN) / gridPos
    velGridStep = (V_MAX - V_MIN) / gridVel
    # Create J matrix
    # row -> position
    # column -> velocity
    J = np.zeros((gridPos + 1, gridVel + 1))
    # Policy matrix
    policy = np.zeros((gridPos+1, gridVel+1))
    u = [0, u_max, -u_max]
    # u = [0, -u_max, u_max]; % <-- '0' comes first, keep in mind that optimal
    # policy is not unique eventhough the cost is unique. If all actions
    # contribute to the same costs, we will select u = 0.
    # Value iteration starts here
    predecessorP = np.zeros((gridPos + 1, gridVel + 1))
    predecessorV = np.zeros((gridPos + 1, gridVel + 1))
    # error = np.zeros(maxHorizon);
    print('value iteration is running...\n')
    for index in range(maxHorizon):

        Jprev = np.copy(J)

        for vIdx in range(nVelStates):
            for pIdx in range(nPosStates):

                # Convert index back to actual value
                p = (pIdx) * posGridStep + P_MIN
                v = (vIdx) * velGridStep + V_MIN

                Jplus1 = np.inf

                for uIdx in u:
                    # Given current input u, find next state information
                    pNext, vNext = mountainCarSim(p, v, u[uIdx])
                    # env.reset()
                    # env.state = [p, v]
                    # next_state, _, _, _ = env.step(uIdx)
                    pNextIdx = snap_to_grid(pNext, P_MIN, P_MAX, gridPos)
                    vNextIdx = snap_to_grid(vNext, V_MIN, V_MAX, gridVel)
                    # pNextIdx, vNextIdx = tuple(discretize(next_state, state_grid))

                    Jplus1_ =  J[pNextIdx, vNextIdx]

                    # if pIdx == 100:
                    #     print()

                    if pNextIdx != gridPos:
                        Jplus1_ = Jplus1_ + 1
                    else:
                        pass
                    # end

                    # Get the smallest one
                    if Jplus1_ < Jplus1:
                        Jplus1 = Jplus1_
                        uMinIdx = uIdx
                        pMinNextIdx = pNextIdx
                        vMinNextIdx = vNextIdx
                    # end

                # end % end for uIdx

                J[pIdx, vIdx] = Jplus1
                policy[pIdx, vIdx] = uMinIdx

                # Store the currrnt optimal node
                predecessorP[pIdx, vIdx] = pMinNextIdx
                predecessorV[pIdx, vIdx] = vMinNextIdx

            # end % end for vIdx
        # end % end for pIdx

        # error[k] = norm(J - Jprev);
        diff = (np.square(J - Jprev)).mean(axis=None)
        # print('episode: %i, error: %f\n', k, error(k))
        print('episode: ', diff)

        if (diff < EPSILON):
            print('converged with error = %f after %i episodes\n')
            # error(k), k);
            break
        # end

    # end % end for k
    # % Resize vector error.
    # error = error(1 : k);
    return J, policy


