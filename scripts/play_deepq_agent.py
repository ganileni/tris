from tris.rules import Match
from tris.agents import RandomAgent
from tris.nn import DeepQLearningAgent

# play a game between a deep-Q agent and a random agent

# initialize agents and match
deepq_learner = DeepQLearningAgent()
random_agent = RandomAgent()
match = Match(deepq_learner, random_agent)
# play loop until end game
match.play()