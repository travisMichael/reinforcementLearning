import numpy as np
from mdp_two_module.mountainCarValueIteration import value_iteration

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
J, policy = value_iteration(uMax, gridSize, 1000)

np.save('policy_2', policy)

# toc
# %% Trace back the optimal policy, for given a certain initial condition
# [XStar, UStar, TStar] = ...
# traceBack(predecessorP, predecessorV, policy, x0, gridSize);
# %% Animation
# visualizeMountainCar(gridPos, XStar, UStar)
# %% Plot errors over iterations.
# figure
# plot(error);
# title('Convergence errors over iterations');
# %% Plot the policy matrix.
# figure
# imagesc(policy)
# title('Policy matrix')
# xlabel('Position');
# ylabel('Velocity');