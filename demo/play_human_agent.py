from tris.rules import Match
from tris.agents import RandomAgent, HumanAgent

# play a game between a human and a random agent

# initialize agents and match
player1, player2 = HumanAgent(), RandomAgent()
match = Match(player1, player2)
# play loop until end game
match.play()