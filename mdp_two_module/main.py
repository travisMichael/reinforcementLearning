import numpy as np
from mdp_two_module.mountainCarValueIteration import value_iteration
from mdp_two_module.extractPolicy import extract_policy_from_values

# %  This is the main file for the simulation.
#     clear all
# close all
# %% Set simulation parameters.
# % Pay attention, there is a minimum grid size here. Otherwise, traceBack
# % function will fail. Since the dynamic equation of the car is:
# %    vNext = v + 0.001 * u - 0.0025 * cos(3 * p);
# % When v(0) = 0, and cos(3p) = 0, u = 1, will result in vNext = 0.001. For
# % this reason, velocity grid should be finer that 0.001.
gridSize = [100, 100]
# %gridSize = [200 200];
gridPos = gridSize[1]
uMax = 1
x0 = [-0.52, 0]
# %x0 = [0.4 0];
# %% Find optimal policy.
# tic
J, policy, scores = value_iteration(uMax, gridSize, 1000, should_score=True)

p_2 = extract_policy_from_values(uMax, gridSize, J)

# np.save('policy_3', policy)
