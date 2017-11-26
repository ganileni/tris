from functions import Match, RandomAgent, QLearningAgent

# play a game between a qlearner and a random agent

# initialize agents and match
qlearner, player2 = QLearningAgent(), RandomAgent()
qlearner.load_agent('prova.pkl')
match = Match(qlearner, player2)
# play loop until end game
match.play()