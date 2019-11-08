import numpy as np
# utility file for the MDP two


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def create_uniform_grid(low, high, bins=(10, 10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid


def value_iteration(environment, state_grid, gamma = 0.99):
    # state_space_size = environment.observation_space.n
    n = len(state_grid[0]) + 1
    V_k = np.zeros(shape=((n, n)))
    # V_k[:,:] = -300

    V_k_previous = np.copy(V_k)
    policy = np.zeros(shape=((n, n)))

    for _ in range(100):
        for i in range(n-1):
            for j in range(n-1):
                state = np.array([state_grid[0][i], state_grid[1][j]])
                V_k[i][j], policy[i][j] = value_max_with_arg_max(state, environment, V_k_previous, state_grid, gamma)

        V_k_previous = np.copy(V_k)

    return V_k, policy


def value_max_with_arg_max(state, env, V_k_previous, state_grid, gamma):
    actions = [0, 1, 2]

    max = 0.0
    best_action = 0
    for action in actions:
        env.state = state
        next_state, reward, _, _ = env.step(action)

        s = discretize(state, state_grid)

        if s[0] == 9 or s[1] == 9:
            continue
        print(reward)
        value = (reward + gamma * V_k_previous[s[0]][s[1]])
        if value > max:
            max = value
            best_action = action

    return max, best_action
