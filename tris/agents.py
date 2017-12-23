import numpy as np
from copy import copy
from tris.functions import pickle_save, pickle_load, softmax, state_from_hash, hash_from_state
from tris.rules import hash_all_states


class Step:
    """Represents the memory of each action the agent took.

    Args:
        state_t: starting state's hash
        action_t: landing state's hash
        coordinates_t: the x,y position of where agent put its mark on the board."""

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
    """Implements the base logic of an agent

    Args:
        default_action_value: the default assigned to all `Action.value`'s in `state_space` generated at initialization."""

    def __init__(self, default_action_value=0):
        # generate all possible game state_space.
        self.default_action_value = default_action_value
        self.state_space = hash_all_states(self.default_action_value)
        self.history = []

    def get_move(self, game_state):
        """get the game as an argument, decide what move to make, save in memory and return the chosen move

        Args:
            game_state: the game state as a 3x3 numpy matrix. 1 is interpreted as 'my symbol', 2 as 'other player's symbol' and 0 as 'empty cell'.

        Returns:
            move_coordinates: the coordinates of where the agent decided to put its mark.
            """
        hashed_state = hash_from_state(game_state)
        move_coordinates, next_state = self.decide_move(hashed_state)
        self._save_move_in_memory(move_coordinates, hashed_state, next_state)
        return move_coordinates

    def endgame(self, reward):
        """Called when the game is finished, the agent saves the game in memory, does cleanup to return to "new game" state, and (maybe) updates policy. (result values of +1 0 -1 stand for win, draw and loss).

        Args:
            reward(float): the reward the agent got at the end of the game.
        """
        raise NotImplementedError

    def _save_move_in_memory(self, move_coordinates, starting_state, next_state):
        """As per the method's name, this adds the last step to the sequence of moves the agent remembers.

        Args:
            move_coordinates((int,int): the coordinates of where the agent put its mark.
            starting_state(int): hash of the starting state (before performing the move)
            next_state(int): the hash of the landing state (after the move has been performed)
        """
        self.history.append(Step(starting_state, next_state, move_coordinates))

    def decide_move(self, hashed_game_state):
        """Decide the move based on current game state.
        Args:
            game_state(int): hashed game state

        Returns:
            move_coordinates: the coordinates of where agent decided to put its mark.
            next_state: the hash of the state coming after the move.
            """
        raise NotImplementedError

    def save_agent(self, path):
        """Save agent to disk.
        Args:
            path: path for the pickle file that will contain the agent. If additional files are needed, they will be in the same directory.
            """
        pickle_save(self.__dict__, path)

    @classmethod
    def load_agent(cls, path):
        """Load an agent from disk.
        Args:
            path: path for the pickle file that contains the agent. If additional files are needed, they will be looked for in the same directory."""
        _dict = pickle_load(path)
        agent = cls()
        for key in _dict:
            setattr(agent, key, _dict[key])
        return agent

    def spawn_self_player(self):
        """Returns an agent with the same methods that can be used for self-play.

        Args:
            [none]

        Returns:
            agent: an agent that can be used for self-play (must be respawned after calling `agent.endgame()`)

        Note: as is clear in the code, the only thing that changes in the returned agent is `agent.history`, which is a new list at a different memory address. `when endgame()` is called on the new spawned agent, it will look into `the new self.history` to memorize or update the policy of the original agent. Note that this trick works only until `agent.endgame()` is called, because `agent.endgame()` should reset `agent.history` when it is done.
        """
        player_copy = copy(self)
        # new history
        player_copy.history = []
        return player_copy


class RandomAgent(BaseAgent):
    """An agent that chooses between possible actions at random. Interface identical to BaseAgent."""

    def decide_move(self, hashed_game_state):
        possible_actions = (self
            .state_space[hashed_game_state]
            .actions)
        choice = np.random.choice(
            list(
                possible_actions.keys()),
            size=1)[0]
        return possible_actions[choice].coordinates, choice

    def endgame(self, reward):
        # do nothing
        pass


class HumanAgent(BaseAgent):
    """Implements the interface for a human to play vs an agent. When `HumanAgent.decide_move()` is called, the program prompts the user a move."""

    def __init__(self):
        super().__init__()
        self.translate = {0: '_',
                          1: 'X',
                          2: 'O'}

    def decide_move(self, hashed_game_state):
        """Print the game state and ask for action."""
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

    def endgame(self, reward):
        comment = {1: 'you won.', -1: 'you lost.', 0: "it's a draw."}
        print(comment[reward])
        pass


class MENACEAgent(BaseAgent):
    """Reproduces the Machine Educable Noughts And Crosses Engine (MENACE) agent, defined in
    Michie, Donald. "Trial and error." Science Survey, Part 2 (1961): 129-145.

    Symmetries of the game are not considered, all possible states are represented.

    For a summary, refer to:
    http://chalkdustmagazine.com/features/menace-machine-educable-noughts-crosses-engine/

    Args:
        beads_n: the number of beads initially for each drawer (corresponding to each state) and each color (corresponding to each move).
        loss: how many beads to add/remove to actions that come before a loss
        win: how many beads to add/remove to actions that come before a win
        draw: how many beads to add/remove to actions that come before a draw
        """

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

    def endgame(self, reward):
        for step in self.history:
            next_state_object = self.state_space[step.state_t].actions[step.action_t]
            # add or remove beads in all state_space visited during the game
            # according to win loss and draw
            next_state_object.value += self.change_beads[reward]
            # number of beads can't go below 0
            if next_state_object.value < 0:
                next_state_object.value = 0
        # clear memory
        self.history = []


class QLearningAgent(BaseAgent):
    """Agent based on a Q-learning rule for learning, and softmax policy.

    Remember, low temperature == low exploration. The parameters of the agent should be changed during training to balance exploration and exploitation.

    Args:
        temperature: temperature of the softmax distribution used to choose next state from the Q-values of alternative possibilities.
        learning_rate: by how much to change the Q-values at each learning iteration.
        discount: by how much to discount future rewards.

    """

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

    def endgame(self, reward):
        inv_history = reversed(self.history)
        reward_multiplier = 1
        step_t2 = next(inv_history)
        for step_t1 in inv_history:
            # find max of Q function on (next step in time)'s possible actions
            max_Q = np.max([_.value for _ in self.state_space[step_t2.state_t].actions.values()])
            # adjust value of Q on current state-action pair accordingly
            # i.e. Q(s, a) = Q(s, a) + learning_rate * (reward_multiplier*r + discount*max_{a'}[Q(s',a')] - Q(s,a))
            self.state_space[step_t1.state_t].actions[step_t1.action_t].value += (
                    self.learning_rate *
                    (reward * reward_multiplier  # only add result if it's final state
                     + self.discount * max_Q  # discount for maxQ of next state in time
                     - self.state_space[step_t1.state_t].actions[step_t1.action_t].value)
            )
            step_t2 = step_t1
            # reward is !=0 only in first step of the game!
            if reward_multiplier: reward_multiplier = 0
        self.history = []
