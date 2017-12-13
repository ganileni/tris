from unittest import TestCase
from unittest.mock import Mock
from tris.rules import Match


class TestMatch(TestCase):
    def setUp(self):
        # mock two agents playing the game.
        # It should end with player1's victory
        self.player1, self.player2 = Mock(), Mock()
        self.player1.get_move = Mock()
        self.player2.get_move = Mock()
        self.player1.get_move.side_effect = [(2, 2), (1, 1), (1, 2), (0, 0)]
        self.player2.get_move.side_effect = [(1, 0), (2, 1), (2, 0)]
        self.match = Match(self.player1, self.player2)
        # let player 1 start
        self.match.who_plays = True

    def test_play(self):
        # result of the game (returned by play())
        # should be +1 for player1's victory
        self.assertEqual(self.match.play(), 1)
        # player1 receives reward +1
        self.player1.endgame.assert_called_with(1)
        # player2 receives reward -1
        self.player2.endgame.assert_called_with(-1)

# this is the game being mock-played:
# Player1
#  (Step: 0 -> 6561
#  crossing: (2, 2),)
# [[ 0.  0.  0.]
#  [ 0.  0.  0.]
#  [ 0.  0.  1.]]
# Player2
# (Step: 13122 -> 13125
#  crossing: (1, 0),)
# [[ 0.  2.  0.]
#  [ 0.  0.  0.]
#  [ 0.  0.  1.]]
# Player1
# (Step: 6567 -> 6648
#  crossing: (1, 1),)
# [[ 0.  2.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]]
# Player2
# (Step: 13287 -> 13530
#  crossing: (2, 1),)
# [[ 0.  2.  0.]
#  [ 0.  1.  2.]
#  [ 0.  0.  1.]]
# Player1
# (Step: 7134 -> 9321
# crossing: (1, 2),)
# [[ 0.  2.  0.]
#  [ 0.  1.  2.]
#  [ 0.  1.  1.]]
# Player2
# (Step: 17904 -> 17913
#  crossing: (2, 0),)
# [[ 0.  2.  2.]
#  [ 0.  1.  2.]
#  [ 0.  1.  1.]]
# Player1
# (Step: 9339 -> 9340
# crossing: (0, 0),)
# [[ 1.  2.  2.]
#  [ 0.  1.  2.]
#  [ 0.  1.  1.]]
# game result is 1.0
