from unittest import TestCase
from functions import Game
import numpy as np


class TestGame(TestCase):
    def test_check_win(self):
        self.states = [
            # test horizontal
            (np.array([[0, 0, 1],
                       [2, 0, 1],
                       [0, 2, 1]]),
             1),
            # test vertical
            (np.array([[0, 0, 1],
                       [2, 0, 1],
                       [2, 2, 2]]),
             2),
            # test diagonal
            (np.array([[1, 0, 1],
                       [2, 1, 0],
                       [0, 2, 1]]),
             1),
            # test reverse diagonal
            (np.array([[0, 0, 2],
                       [2, 2, 1],
                       [2, 2, 1]]),
             2),
            # test draw
            (np.array([[1, 2, 1],
                       [2, 2, 1],
                       [2, 1, 2]]),
             0),
            # test game not ended
            (np.array([[0, 0, 1],
                       [2, 0, 0],
                       [0, 2, 1]]),
             None),
        ]
        self.game = Game()
        for state,expected_result in self.states:
            self.game.state = state
            self.assertEqual(self.game.check_win(), expected_result, msg = repr(state))

    def test_player_move(self):
        self.game = Game()
        self.game.player_move(x=0,y=0,player=1)
        self.assertEqual(self.game.state[0,0],1)

    def test_check_win(self):
        self.game = Game()
        self.game.state = (np.array([[0, 0, 1],
                                    [2, 0, 1],
                                    [0, 2, 1]]))
        reverse_state = (np.array([[0, 0, 2],
                                    [1, 0, 2],
                                    [0, 1, 2]]))
        self.assertListEqual(self.game.reverse_state.flatten().tolist(),
                             reverse_state.flatten().tolist())
