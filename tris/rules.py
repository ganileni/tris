import numpy as np
from tris.constants import x_coordinates, y_coordinates, starting_state, max_state_hash
from tris.functions import state_from_hash, hash_from_state

# maps 2 to -1 for changing the results representation convention
two_to_minusone = {0: 0, 1: 1, 2: -1}


def eval_possible_actions(state, default_value=0):
    """given a state, evaluate and return a dict containing
    all possible actions that can be taken from that state."""
    actions = dict()
    for y in y_coordinates:
        for x in x_coordinates:
            if not state[y, x]:  # 0 means empty
                new_state = state.copy()
                new_state[y, x] = 1  # 1 means "mine", 2 means "the opponent's"
                # hash of new move mapped to tuple of coordinates of change
                actions[hash_from_state(new_state)] = Action(x, y, value=default_value)
    return actions


class StateSpace:
    """Behaves like a defaultdict that, when accessed at a new key,
    adds a GameState to itself. Keys are supposed to be
    hashes of possible game states.

    This is based on the idea that the majority of hashes
    between 0 and 3**9 contain illegal states, and therefore
    creating state spaces with the hash_all_states() function
    is a waste of memory and search time.

    Turns out that because it's not a builtin,
    just using a dict() is faster, though. So it won't be used.
    """

    def __init__(self, default_action_value=0):
        self.default_action_value = default_action_value
        self._states = dict()

    def __getitem__(self, item):
        try:
            return self._states[item]
        # if hash of state is not in dict, generate a GameState
        except KeyError:
            self._states[item] = GameState(id_hash=item, default_action_value=self.default_action_value)
            return self._states[item]

    def __setitem__(self, key, value):
        self._states[key] = value

    def __iter__(self):
        return self._states.__iter__()


def is_illegal_state(hashed_state):
    """because no player can do two consecutive moves,
    all states where the number of Xs and Os differs
    by more than 1 is illegal. This returns True if
    the state hashed by `hashed_state` is illegal."""
    state = state_from_hash(hashed_state)
    # illegal iff abs(num(X) - num(O)) > 1
    # X is 1 O is 2 (or viceversa)
    return np.abs((state == 1).sum() - (state == 2).sum()) > 1


def hash_all_states(default_action_value=0):
    """produce a dict that maps all possible hashes
    to all possible game state_space.
    symmetries are not taken into account in this implementation."""

    all_states = dict()
    for hash_id in range(max_state_hash):
        # if game state is illegal, shouldn't be added
        if is_illegal_state(hash_id):
            continue
        all_states[hash_id] = GameState(hash_id, default_action_value)
    return all_states


class Action:
    """represents an action taken by an agent.
    x,y are the coordinates where the agent
    puts their cross (or naught).
    `value` might be used to store information
    about the action, according to
    the agent's implementation"""

    def __init__(self, x, y, value=0):
        self.coordinates = (x, y)
        # this is supposed to store policy expectation value
        self.value = value


class GameState:
    """represents a state of the game. contains the state as a 3x3 np.array
    the hash of the state, and a dict of all possible actions that can be taken.
    default_value is the value argument passed to Action.__init__()
    when calculating possible actions"""

    def __init__(self, id_hash, default_action_value=0):
        self.state = state_from_hash(id_hash)
        self.hash = id_hash
        self.actions = eval_possible_actions(self.state, default_action_value)


class Game:
    """implements the rules of tic tac toe"""

    def __init__(self):
        self.state = starting_state()

    @property
    def reverse_state(self):
        """returns game state with 1 and 2 swapped.
        lazily evaluated."""
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
    """implements the logic to interface two players
    according to the rules of tic-tac-toe"""

    def __init__(self, player1, player2):
        self.game = Game()
        self.player1 = player1
        self.player2 = player2
        self.result = None
        self.history = []
        # a random bool, determines who plays first
        # (False -> player1)
        # note you can alter this by changing
        # self.who_plays before calling Match.play()
        self.who_plays = np.random.rand() >= .5
        # dict with bool keys. this will alternate between players.
        self.player_actions = {False: self._move_player1,
                               True: self._move_player2}

    def _move_player1(self):
        """ask for player1's move and apply it to the game"""
        move = self.player1.get_move(game_state=self.game.state)
        self.game.player_move(player=1, x=move[0], y=move[1])

    def _move_player2(self):
        # reverse the state because each agent sees 1 as "me" and 2 as "opponent"
        move = self.player2.get_move(game_state=self.game.reverse_state)
        self.game.player_move(player=2, x=move[0], y=move[1])

    def play(self):
        """play in a loop until game ends,
        then tell result to agents
        finally return it: 1=player1 win, -1=player2 winn, 0=draw.
        """
        # while game is not finished
        while self.result is None:
            self._next_step()
        # when the game is done
        return self._assign_rewards()

    def _assign_rewards(self):
        if self.result:  # if not draw
            if self.result == 1:  # if win player1
                self.player1.endgame(1)
                self.player2.endgame(-1)
            else:  # if win player2
                self.player1.endgame(-1)
                self.player2.endgame(1)
        else:  # if draw
            self.player1.endgame(0)
            self.player2.endgame(0)
        # change 2's to -1's
        return two_to_minusone[self.result]

    def _next_step(self):
        """play next player move and check if game is finished"""
        # switch to next player
        self.who_plays = not self.who_plays
        # ask player to move
        self.player_actions[self.who_plays]()
        # check for game end
        self.result = self.game.check_win()
        # save in history
        self.history.append(self.game.state.copy())
