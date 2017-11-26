import pickle

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
    adapted from: https://nolanbconaway.github.io/blog/2017/softmax-numpy
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
    # subtract the max for numerical stability & expontentiate
    y = np.exp(y - np.max(y))
    # take the sum & divide elementwise
    p = y / y.sum()
    return p


def ternary(n: int):
    """return the representation of decimal int n
    in base 3, as a length 9 list of ints"""
    if not n:
        return ''.zfill(9)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    return nums + [0] * (9 - len(nums))


def state_from_hash(hash_id: int):
    """takes in a hash and outputs a 3x3 numpy array (a game state)
    it's the inverse of hash_from_state (there is a bijection)"""
    hash_id = (x for x in ternary(hash_id))
    state = starting_state()
    for y in y_coordinates:
        for x in x_coordinates:
            state[x, y] = next(hash_id)
    return state.T


def hash_from_state(state):
    """takes in a 3x3 numpy array (a game state)
    and hashes it with powers of 3. returns an int.
    it's the inverse of the state_from_hash"""
    # state must be 3x3 np array
    return sum([cell * (3 ** power) for power, cell in enumerate(state.flatten())])