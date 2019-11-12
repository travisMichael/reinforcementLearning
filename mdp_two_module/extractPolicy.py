import numpy as np
from mdp_two_module.mountainCarSimulation import mountainCarSim
from discrete.snapToGrid import snap_to_grid


def extract_policy_from_values(u_max, gridSize, J):
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

    u = [0, u_max, -u_max]
    policy = np.zeros((gridPos+1, gridVel+1))

    predecessorP = np.zeros((gridPos + 1, gridVel + 1))
    predecessorV = np.zeros((gridPos + 1, gridVel + 1))
    # error = np.zeros(maxHorizon);
    print('policy iteration is running...\n')
    # Jprev = np.copy(J)

    for vIdx in range(nVelStates):
        for pIdx in range(nPosStates):

            # Convert index back to actual value
            p = (pIdx) * posGridStep + P_MIN
            v = (vIdx) * velGridStep + V_MIN

            Jplus1 = np.inf

            for uIdx in u:
                # Given current input u, find next state information
                pNext, vNext = mountainCarSim(p, v, u[uIdx])
                pNextIdx = snap_to_grid(pNext, P_MIN, P_MAX, gridPos)
                vNextIdx = snap_to_grid(vNext, V_MIN, V_MAX, gridVel)

                Jplus1_ =  J[pNextIdx, vNextIdx]


                if pNextIdx != gridPos:
                    Jplus1_ = 0.99 * Jplus1_ + 1
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

            policy[pIdx, vIdx] = uMinIdx

            # Store the currrnt optimal node
            predecessorP[pIdx, vIdx] = pMinNextIdx
            predecessorV[pIdx, vIdx] = vMinNextIdx

    return policy
