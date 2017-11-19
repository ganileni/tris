import numpy as np
from collections import defaultdict
from tqdm import tqdm

# http://www.cs.dartmouth.edu/~lorenzo/teaching/cs134/Archive/Spring2009/final/PengTao/final_report.pdf

x_coordinates = [0, 1, 2]
y_coordinates = x_coordinates
starting_state = lambda: np.zeros((3, 3))


def hash_from_state(state):
    # state must be 3x3 np array
    return sum([cell * (3 ** power) for power, cell in enumerate(state.flatten())])


def ternary(n):
    if not n:
        return ''.zfill(9)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    return nums + [0] * (9 - len(nums))
    # return ''.join(nums)[:,:,-1].zfill(9)[:,:,-1]


# thank god there is a bijection
# we'll use that to generate all possible states
def state_from_hash(hash_id):
    hash_id = (x for x in ternary(hash_id))
    state = starting_state()
    for y in y_coordinates:
        for x in x_coordinates:
            state[x, y] = next(hash_id)
    return state.T


def eval_possible_actions(state):
    actions = defaultdict(float)
    for y in y_coordinates:
        for x in x_coordinates:
            if not state[y, x]:  # 0 means empty
                new_state = state.copy()
                new_state[y, x] = 1  # 1 means "mine", 2 means "the opponent's"
                actions[hash_from_state(new_state)]
    return actions


class GameState:
    def __init__(self, id_hash):
        self.state = state_from_hash(id_hash)
        # id for hashing
        self.hash = id_hash
        self.actions = eval_possible_actions(self.state)


class Game:
    def __init__(self):
        self.state = starting_state()

    def player_move(self, player: int, x: int, y: int):
        # make sure xy is empty and player is either 1 or 2
        assert not self.state[y, x] and player < 3
        self.state[y, x] = player

    def check_win(self):
        # check verticals
        for x in x_coordinates:
            # for each column: if all values the same and !=0
            if len(set(self.state[:, x])) == 1 and self.state[0, x] != 0:
                return self.state[0, x]
        # check horizontals
        for y in y_coordinates:
            if len(set(self.state[y, :])) == 1 and self.state[y, 0] != 0:
                return self.state[y, 0]
        # check diagonals
        if len(set(np.diag(self.state))) == 1 and self.state[0, 0] != 0:
            return self.state[0, 0]
        if len(set(np.diag(np.fliplr(self.state)))) == 1 and self.state[0, 2] != 0:
            return self.state[0, 2]
        #if there is no win and no space left, it's a draw
        if 0 not in self.state:
            return 0
        return None


max_state_hash = 3 ** 9
