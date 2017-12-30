import pickle
from copy import copy
import numpy as np
from tris.constants import starting_state, y_coordinates, x_coordinates


def pickle_save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def pickle_load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def softmax(X, temperature=1.0):
    """
    adapted from: https://nolanbconaway.github.io/blog/2017/softmax-numpy.
    credit where is due.

    Compute the softmax of each element of X.

    Parameters
    ----------
    X: 1D-Array of numeric types.
    temperature (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    Returns an array the same size as X. The result will sum to 1.
    """
    # div by temperature
    y = np.array(X) / float(temperature)
    # subtract the max for numerical stability & exponentiate
    y = np.exp(y - np.max(y))
    # take the sum & divide elementwise
    p = y / y.sum()
    return p


def ternary(n: int):
    """return the representation of decimal int n in base 3, as a length 9 list of ints

    Args:
        n: int number. must be <= 3**9
    Returns:
        base3: the representation of n in base 3, as a list of 9 ints.
        """
    if not n:
        return ''.zfill(9)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    return nums + [0] * (9 - len(nums))


def state_from_hash(hash_id: int):
    """takes in a hash and outputs a 3x3 numpy array (a game state) it's the inverse of hash_from_state (there is a
    bijection)

    Args:
        hash_id: the hash of a game state. must be an int between 0 and 3**9.
    Returns:
        state: the game state as a 3x3 np.array of ints (either 0, 1 or 2)
    """
    hash_id = (x for x in ternary(hash_id))
    state = starting_state()
    for y in y_coordinates:
        for x in x_coordinates:
            state[x, y] = next(hash_id)
    return state.T


def chunkit(seq, num):
    """just chunks a list `num` parts

    Args:
        seq: the sequence to be chopped
        num: the number of chunks requested
    Returns:
        out: a list of lists. it is of len `num`.
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def hash_from_state(state):
    """takes in a 3x3 numpy array (a game state) and hashes it with powers of 3. returns an int. it's the inverse of
    the state_from_hash

    Args:
        state: a 3x3 numpy array representing a tic-tac-toe game state. must contain only 0, 1, or 2's.
    Returns:
        hash: the hash corresponding to the state: an in between 0 and 3**9.
        """
    # state must be 3x3 np array, with values either 0, 1, or 2.
    return sum([cell * (3 ** power) for power, cell in enumerate(state.flatten())])


def count_visited_actions(agent, default_value):
    """to check the percent of states and actions that changed value in an agent after some training"""
    all_actions = 0
    visited_actions = 0
    visited_states = 0
    for state in agent.state_space:
        state_is_visited = 0
        state_actions = agent.state_space[state].actions
        all_actions += len(state_actions.keys())
        for action in state_actions:
            if state_actions[action].value != default_value:
                visited_actions += 1
                state_is_visited += 1
        if state_is_visited: visited_states += 1
    return visited_states / len(agent.state_space), \
           visited_actions / all_actions


def time_average(vector, window=100):
    """compute time averages for timeseries `vector`. `vector` must be a list/np.array of numbers, `window` is the
    averaging window. returns a list of numbers.

    Args:
        vector: a list of numbers
        window: the window of time over which to average
    Returns:
        avgs: a list containing the time averages.
        """
    # TODO: rewrite so that it applies an arbitrary function to rolling window
    avgs = []
    vector_copy = copy(vector)
    while vector_copy:
        avgs.append(np.mean(vector_copy[:window]))
        vector_copy[:window] = []
    return avgs
