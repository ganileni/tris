import numpy as np
from collections import defaultdict
from tqdm import tqdm

# http://www.cs.dartmouth.edu/~lorenzo/teaching/cs134/Archive/Spring2009/final/PengTao/final_report.pdf

x_coordinates = [0, 1, 2]
y_coordinates = x_coordinates
starting_state = lambda: np.zeros((3, 3))
max_state_hash = 3 ** 9


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


def hash_all_states():
    all_states = dict()
    for hash_id in range(max_state_hash):
        all_states[hash_id] = GameState(hash_id)
    return all_states


# thank god there is a bijection
# we'll use that to generate all possible states
def state_from_hash(hash_id):
    hash_id = (x for x in ternary(hash_id))
    state = starting_state()
    for y in y_coordinates:
        for x in x_coordinates:
            state[x, y] = next(hash_id)
    return state.T


class Action:
    def __init__(self, x, y, value=0):
        self.coordinates = (x, y)
        # this is supposed to store policy expectation value
        self.value = value


def eval_possible_actions(state):
    actions = dict()
    for y in y_coordinates:
        for x in x_coordinates:
            if not state[y, x]:  # 0 means empty
                new_state = state.copy()
                new_state[y, x] = 1  # 1 means "mine", 2 means "the opponent's"
                # hash of new move mapped to tuple of coordinates of change
                actions[hash_from_state(new_state)] = Action(x, y)
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

    @property
    def reverse_state(self):
        """returns game state with 1 and 2 swapped"""
        rs = self.state.copy()
        ones, twos = rs == 1, rs == 2
        rs[ones], rs[twos] = 2, 1
        return rs

    def player_move(self, player: int, x: int, y: int):
        """applies the move of `player` at coordinates `x`,`y`"""
        # make sure xy is empty and player is either 1 or 2
        assert not self.state[y, x] and player < 3
        self.state[y, x] = player

    def check_win(self):
        """returns 1 or 2 if player 1 or 2 won respectively
        0 if draw, None if game not ended."""
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
        # if there is no win and no space left, it's a draw
        if 0 not in self.state:
            return 0
        return None


class Match:
    def __init__(self, player1, player2):
        self.game = Game()
        self.player1 = player1
        self.player2 = player2
        self.result = None
        self.history = []
        # a random bool
        self.who_plays = np.random.rand() >= .5
        # better to implement this with functools!
        # assign to random bools in a dict
        # make sure implementation is not buggy and youre not
        # confusing player 1 and player 2
        self.player_actions = {self.who_plays: self.move_player1,
                               not self.who_plays: self.move_player2}

    def move_player1(self):
        move = self.player1.move(self.game.state)
        self.game.player_move(player=1, x=move[0], y=move[1])

    def move_player2(self):
        # reverse the state because each agent sees 1 as "me" and 2 as "opponent"
        move = self.player2.move(self.game.reverse_state)
        self.game.player_move(player=2, x=move[0], y=move[1])

    def play(self):
        while self.result is None:
            self.who_plays = not self.who_plays
            self.player_actions[self.who_plays]()
            self.result = self.game.check_win()
            self.history.append(self.game.state.copy())
        # when the game is done, assign scores
        if self.result:
            if self.result == 1:
                self.player1.endgame(1)
                self.player2.endgame(-1)
            else:
                self.player1.endgame(-1)
                self.player2.endgame(1)
        else:
            self.player1.endgame(0)
            self.player2.endgame(0)
        return self.result


class BaseAgent:
    """implements the base logic of an agent"""

    def __init__(self):
        self.states_space = hash_all_states()
        self.state_history = []
        self.move_history = []
        # generate all possible game states

    def move(self, game_state):
        """get the game as an argument, decide what move to make,
        save in memory and return the chosen move"""
        hashed_state = hash_from_state(game_state)
        move = self.decide_move(hashed_state)
        self.save_in_memory(move, hashed_state)
        return move

    def endgame(self, result):
        """when game finished, calculate new policy from result
        result values of +1 0 -1 stand for win, draw and loss"""
        raise NotImplementedError

    def save_in_memory(self, move, hashed_state):
        """remember the sequence of moves from which to calculate
        the new policy"""
        self.state_history.append(hashed_state)
        self.move_history.append(move)

    def decide_move(self, game_state):
        """decide next move based on game state"""
        raise NotImplementedError

    def save_agent(self, path):
        """save agent to disk"""
        raise NotImplementedError

    def load_agent(self, path):
        """load an agent from disk"""
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """an agent that chooses between possible actions at random"""

    def decide_move(self, hashed_game_state):
        possible_actions = (self
                            .states_space[hashed_game_state]
                            .actions)
        return possible_actions[np.random.choice(list(possible_actions.keys()),
                                                 size=1)[0]].coordinates

    def endgame(self, result):
        pass


class HumanAgent(BaseAgent):
    """implements the interface for a human to play vs an agent"""

    def __init__(self):
        super().__init__()
        self.translate = {0: '_',
                          1: 'X',
                          2: 'O'}

    def decide_move(self, hashed_game_state):
        state = state_from_hash(hashed_game_state)
        print("\t0\t1\t2\tx")
        for row, rowname in zip(state, range(3)):
            print('\t'.join([str(rowname)] + [self.translate[_] for _ in row]))
        print('y\n')
        # get input from player
        x = int(input('What x?'))
        y = int(input('What y?'))
        return (x, y)

        raise NotImplementedError

    def endgame(self, result):
        # remark who won
        pass


class MENACEAgent(BaseAgent):
    """reproduces MENACE agent, defined in
    Michie, Donald. "Trial and error." Science Survey, Part 2 (1961): 129-145."""

    def __init__(self, beads_n=100, loss=-2, win=+2, draw=-1):
        super().__init__()
        self.next_state_history = []
        self.change_beads = dict()
        self.change_beads[1], self.change_beads[-1], self.change_beads[0] = win, loss, draw
        self.win, self.loss, self.draw = win, loss, draw
        for key in self.states_space:
            possible_actions = self.states_space[key].actions
            for action in possible_actions:
                possible_actions[action].value = beads_n

    def decide_move(self, hashed_state):
        # retrieve GameState object
        current_state = self.states_space[hashed_state]
        actions = [_ for _ in current_state.actions]
        # number of beads per action will be proportional to probability of choice
        beads = np.array([current_state.actions[_].value for _ in actions])
        # normalize probabilities
        beads = beads / beads.sum()
        choice = np.random.choice(actions, size=1, p=beads)[0]
        self.next_state_history.append(choice)
        return current_state.actions[choice].coordinates

    def endgame(self, result):
        for state, next_state in zip(self.state_history, self.next_state_history):
            next_state_object = self.states_space[state].actions[next_state]
            # add or remove beads in all states visited during the game
            ## according to win loss and draw
            next_state_object.value += self.change_beads[result]
            # number of beads can't go below 0
            if next_state_object.value < 0:
                next_state_object.value = 0
        # clear memory
        self.state_history, self.next_state_history = [], []
