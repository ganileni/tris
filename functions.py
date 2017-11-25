import numpy as np
import pickle

x_coordinates = [0, 1, 2]
y_coordinates = x_coordinates
starting_state = lambda: np.zeros((3, 3))
max_state_hash = 3 ** 9


def pickle_save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def pickle_load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def ternary(n):
    if not n:
        return ''.zfill(9)
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    return nums + [0] * (9 - len(nums))


def state_from_hash(hash_id):
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


def eval_possible_actions(state):
    """given a state, evaluate and return a dict containing
    all possible actions that can be taken from that state."""
    actions = dict()
    for y in y_coordinates:
        for x in x_coordinates:
            if not state[y, x]:  # 0 means empty
                new_state = state.copy()
                new_state[y, x] = 1  # 1 means "mine", 2 means "the opponent's"
                # hash of new move mapped to tuple of coordinates of change
                actions[hash_from_state(new_state)] = Action(x, y)
    return actions


class Action:
    """represents an action taken by an agent.
    x,y are the coordinates where the agent
    puts their cross (or naught).
    value might be used to store information
    about the action, according to
    the agent's implementation"""

    def __init__(self, x, y, value=0):
        self.coordinates = (x, y)
        # this is supposed to store policy expectation value
        self.value = value


def hash_all_states():
    """produce a dict that maps all possible hashes
    to all possible game state_space.
    symmetries are not taken into account in this implementation."""
    all_states = dict()
    for hash_id in range(max_state_hash):
        all_states[hash_id] = GameState(hash_id)
    return all_states


class GameState:
    """represents a state of the game. contains the state as a 3x3 np.array
    the hash of the state, and a dict of all possible actions that can be taken."""

    def __init__(self, id_hash):
        self.state = state_from_hash(id_hash)
        self.hash = id_hash
        self.actions = eval_possible_actions(self.state)


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
        # a random bool
        self.who_plays = np.random.rand() >= .5
        # maybe better to implement this with functools?
        # dict with bool keys. this will alternate between players.
        self.player_actions = {self.who_plays: self.move_player1,
                               not self.who_plays: self.move_player2}

    def move_player1(self):
        """ask for player1's move and apply it to the game"""
        move = self.player1.move(self.game.state)
        self.game.player_move(player=1, x=move[0], y=move[1])

    def move_player2(self):
        # reverse the state because each agent sees 1 as "me" and 2 as "opponent"
        move = self.player2.move(self.game.reverse_state)
        self.game.player_move(player=2, x=move[0], y=move[1])

    def play(self):
        """play in a loop until game ends,
        then tell result to agents
        finally return it: 1=win, -1=loss, 0=draw.
        """
        # while game is not finished
        while self.result is None:
            # switch to next player
            self.who_plays = not self.who_plays
            # ask player to move
            self.player_actions[self.who_plays]()
            # check for game end
            self.result = self.game.check_win()
            self.history.append(self.game.state.copy())
        # when the game is done, assign rewards
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
        return self.result


class Step:
    def __init__(self, state_t, action_t, coordinates_t):
        self.state_t = state_t
        # the action is actually represented bu the hash of
        # the next state
        self.action_t = action_t
        self.coordinates_t = coordinates_t

    def __repr__(self):
        return ("Step: "
                + str(int(self.state_t))
                + " -> "
                + str(int(self.action_t))
                + "\n crossing: " + str(self.coordinates_t))


class BaseAgent:
    """implements the base logic of an agent"""

    def __init__(self):
        # generate all possible game state_space.
        self.state_space = hash_all_states()
        self.history = []

    def move(self, game_state):
        """get the game as an argument, decide what move to make,
        save in memory and return the chosen move"""
        hashed_state = hash_from_state(game_state)
        move_coordinates, next_state = self.decide_move(hashed_state)
        self.save_in_memory(move_coordinates, hashed_state, next_state)
        return move_coordinates

    def endgame(self, result):
        """when game finished, calculate new policy from result
        and cleanup to return to "new game" agent state
        (result values of +1 0 -1 stand for win, draw and loss)"""
        raise NotImplementedError

    def save_in_memory(self, move_coordinates, hashed_state, next_state):
        """remember the sequence of moves from which to calculate
        the new policy"""
        self.history.append(Step(hashed_state, next_state, move_coordinates))

    def decide_move(self, game_state):
        """decide and then return the next state and next move based on current game state"""
        raise NotImplementedError

    def save_agent(self, path):
        """save agent to disk"""
        pickle_save(self.__dict__, path)

    def load_agent(self, path):
        """load an agent from disk"""
        _dict = pickle_load(path)
        for key in _dict:
            setattr(self, key, _dict[key])


class RandomAgent(BaseAgent):
    """an agent that chooses between possible actions at random"""

    def decide_move(self, hashed_game_state):
        possible_actions = (self
                            .state_space[hashed_game_state]
                            .actions)
        action_taken = np.random.choice(
            list(
                possible_actions.keys()),
            size=1)[0]
        return possible_actions[action_taken].coordinates, action_taken

    def endgame(self, result):
        # do nothing
        pass


class HumanAgent(BaseAgent):
    """implements the interface for a human to play vs an agent"""

    def __init__(self):
        super().__init__()
        self.translate = {0: '_',
                          1: 'X',
                          2: 'O'}

    def decide_move(self, hashed_game_state):
        """print the game state and ask for action"""
        state = state_from_hash(hashed_game_state)
        print("\n\n\t0\t1\t2\tx")
        for row, rowname in zip(state, range(3)):
            print('\t'.join([str(rowname)] + [self.translate[_] for _ in row]))
        print('y\n')
        # get input from player, loop until a legal move is input
        move_unknown = True
        while move_unknown:
            x = int(input('What x? '))
            y = int(input('What y? '))
            if not state[y, x]:
                move_unknown = False
            else:
                print("illegal move!")
        return x, y

        raise NotImplementedError

    def endgame(self, result):
        comment = {1: 'you won.', -1: 'you lost.', 0: "it's a draw."}
        print(comment[result])
        pass


class MENACEAgent(BaseAgent):
    """reproduces MENACE agent, defined in
    Michie, Donald. "Trial and error." Science Survey, Part 2 (1961): 129-145."""

    def __init__(self, beads_n=100, loss=-2, win=+2, draw=-1):
        """parameters:
        beads_n == how many starting beads for each action
        (loss,win,draw) == how many beads to add for each action in case of (loss,win,draw)
        """
        super().__init__()
        self.change_beads = dict()
        self.change_beads[1], self.change_beads[-1], self.change_beads[0] = win, loss, draw
        self.win, self.loss, self.draw = win, loss, draw
        for key in self.state_space:
            possible_actions = self.state_space[key].actions
            for action in possible_actions:
                possible_actions[action].value = beads_n

    def decide_move(self, hashed_state):
        # retrieve GameState object
        current_state = self.state_space[hashed_state]
        actions = [_ for _ in current_state.actions]
        # number of beads per action will be proportional to probability of choice
        beads = np.array([current_state.actions[_].value for _ in actions])
        # normalize probabilities
        beads = beads / beads.sum()
        choice = np.random.choice(actions, size=1, p=beads)[0]
        return current_state.actions[choice].coordinates, choice

    def endgame(self, result):
        for step in self.history:
            next_state_object = self.state_space[step.state_t].actions[step.action_t]
            # add or remove beads in all state_space visited during the game
            # according to win loss and draw
            next_state_object.value += self.change_beads[result]
            # number of beads can't go below 0
            if next_state_object.value < 0:
                next_state_object.value = 0
        # clear memory
        self.history = []


class Q_function():
    def __init__(self, state_space):
        self.state_space = state_space

    def __call__(self, state, action):
        return self.state_space[state].actions[action]


class QLearningAgent(BaseAgent):
    def __init__(self, temperature=.5, learning_rate=.1, discount=.9):
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.discount = discount
        super().__init__()
        self.Q = Q_function(self.state_space)

    def decide_move(self, game_state):
        pass

    def endgame(self, result):
        pass
