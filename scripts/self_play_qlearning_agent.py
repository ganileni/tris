from tris.rules import Match
from tris.agents import QLearningAgent

# play a game between a qlearner and a random agent

# initialize agents and match
qlearner = QLearningAgent()
qlearner.spawn_self_player()
match = Match(qlearner, qlearner.spawn_self_player())
# play loop until end game
match.play()
# you ned to spawn a new self player at each match
match = Match(qlearner, qlearner.spawn_self_player())
match.play()
