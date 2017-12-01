#just check that you can play a full match with each implemented agent without errors


from unittest import TestCase
from tris.nn import DeepQLearningAgent
from tris.rules import Match
from tris.agents import RandomAgent, MENACEAgent, QLearningAgent


class TestGame(TestCase):
    def test_all_agents(self):
        for agent in [RandomAgent, MENACEAgent, QLearningAgent, DeepQLearningAgent]:
            # initialize agents and match
            player1 = agent()
            player2 = RandomAgent()
            match = Match(player1, player2)
            # play loop until end game
            match.play()
            self.assertTrue(True)
