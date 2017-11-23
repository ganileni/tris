from functions import Match, RandomAgent, state_from_hash

# play a game between two random agents, print the moves and the result

# initialize agents and match
player1, player2 = RandomAgent(), RandomAgent()
match = Match(player1, player2)
# play loop until end game
match.play()
print('couples of moves')
print(list(zip(player1.move_history, player2.move_history)))
print('sequence of states')
for _ in match.history:
    print(_)
print('game result is', match.result)
