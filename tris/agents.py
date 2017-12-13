import numpy as np
from copy import copy
from tris.functions import pickle_save, pickle_load, softmax, state_from_hash, hash_from_state
from tris.rules import hash_all_states


class Step:
    """represents the memory of each action the agent took.
    state_t is original state's hash
    action_t is landing state's hash
    coordinates_t are the x,y position of where agent put its mark on the board."""
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

    def __init__(self, default_action_value=0):
        """default_action_value is the default assigned to all Action.value
        that are automatically generated when accessing a new state in self.state_space"""
        # generate all possible game state_space.
        self.default_action_value = default_action_value
        self.state_space =  hash_all_states(self.default_action_value)
        self.history = []

    def get_move(self, game_state):
        """get the game as an argument, decide what move to make,
        save in memory and return the chosen move"""
        hashed_state = hash_from_state(game_state)
        move_coordinates, next_state = self.decide_move(hashed_state)
        self._save_move_in_memory(move_coordinates, hashed_state, next_state)
        return move_coordinates

    def endgame(self, result):
        """when game finished,
        and cleanup to return to "new game" agent state
        and (maybe) calculate new policy from result
        (result values of +1 0 -1 stand for win, draw and loss)"""
        raise NotImplementedError

    def _save_move_in_memory(self, move_coordinates, hashed_state, next_state):
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

    def spawn_self_player(self):
        """returns an agent with the same methods that can be used for self-play.
        the only thing that changes in the returned agent is self.history, which
        is a list at a different memory address. when endgame() is called on the
        new player, it will look into `the new self.history`.
        note that this trick works only until endgame() is called on the
        spawned player!"""
        player_copy = copy(self)
        # new history
        player_copy.history = []
        return player_copy


class RandomAgent(BaseAgent):
    """an agent that chooses between possible actions at random"""

    def decide_move(self, hashed_game_state):
        possible_actions = (self
                            .state_space[hashed_game_state]
                            .actions)
        choice = np.random.choice(
                list(
                        possible_actions.keys()),
                size=1)[0]
        return possible_actions[choice].coordinates, choice

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
        return (x, y), None

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
        super().__init__(beads_n)
        self.change_beads = dict()
        self.change_beads[1], self.change_beads[-1], self.change_beads[0] = win, loss, draw
        self.win, self.loss, self.draw = win, loss, draw
        # for key in self.state_space:
        #     possible_actions = self.state_space[key].actions
        #     for action in possible_actions:
        #         possible_actions[action].value = beads_n

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


class QLearningAgent(BaseAgent):
    """agent based on a q-learning rule for learning
    and softmax policy. low temperature == low exploration"""

    def __init__(self, temperature=1, learning_rate=.1, discount=.9):
        super().__init__()
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.discount = discount

    def decide_move(self, game_state):
        possible_actions = self.state_space[game_state].actions
        # find Q-values for possible actions
        Q_values = [possible_actions[_].value for _ in possible_actions.keys()]
        choice = np.random.choice(list(possible_actions.keys()),
                                  size=1,
                                  p=softmax(Q_values, self.temperature))[0]
        return possible_actions[choice].coordinates, choice

    def endgame(self, result):
        inv_history = reversed(self.history)
        reward_multiplier = 1
        step_t2 = next(inv_history)
        for step_t1 in inv_history:
            # find max of Q function on (next step in time)'s possible actions
            max_Q = np.max([_.value for _ in self.state_space[step_t2.state_t].actions.values()])
            # adjust value of Q on current state-action pair accordingly
            self.state_space[step_t1.state_t].actions[step_t1.action_t].value += (
                self.learning_rate *
                (result * reward_multiplier  # only add result if it's final state
                 + self.discount * max_Q  # discount for maxQ of next state in time
                 - self.state_space[step_t1.state_t].actions[step_t1.action_t].value)
            )
            step_t2 = step_t1
            # reward is !=0 only in first step of the game!
            if reward_multiplier: reward_multiplier = 0
        self.history = []


