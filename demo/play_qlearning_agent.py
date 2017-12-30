"""this script demos how to initialize an agent and let it play against another agent."""

from tris.rules import Match
from tris.agents import RandomAgent, QLearningAgent

# play a game between a qlearner and a random agent

# initialize agents and match
qlearner, player2 = QLearningAgent(), RandomAgent()
match = Match(qlearner, player2)
# play loop until end game
match.play()