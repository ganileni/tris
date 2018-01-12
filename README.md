# Tris

Tris is a Python implementation of several reinforcement learning algoritms to play tic-tac-toe. At the moment it comprises: Q-learning, deep-Q learning and a MENACE algorithm. I plan to implement deep-Q learning with monte-carlo tree search and a Q-function approximator with gaussian processes.

If you want to have a quick play against one of the agents, just start [`scripts/human_play.py`](scripts/human_play.py) like this:
```
cd scripts/
python human_play.py
```

## Motivation

As an exercise, I decided to try and implement several reinforcement learning algorithms. Everything is implemented from scratch from builtins and numpy, except for neural networks, which are based on tensorflow. By the way, tris is the Italian name for tic-tac-toe, also known as naughts and crosses.


## Code Example
```
from tris.rules import Match
from tris.agents import RandomAgent, QLearningAgent
from tris.functions import time_average
import matplotlib.pyplot as plt

# generate two agents
qlearner, player2 = QLearningAgent(), RandomAgent()

match_results=[]
for _ in range(10000):
    # create a new match
    match = Match(qlearner, player2)
    # play match until end,
    # QLearningAgent learns automatically at the end.
    match_results.append(match.play())

# check that agent's performance increases
plt.plot(time_average(match_results))
plt.show()
```
 
Code examples are available in [`demo/`](demo/) and [`scripts/`](scripts/). `scripts/` contains the machinery needed to run the `human_play.py` script, which involves loading agents from disk, training them if they have never been trained, and actually playing against a human.

## Installation

The code has been tested with python 3.6.
Just pull the code with

`git clone git@github.com:ganileni/tris.git`

and install the requirements:

`cd tris`

`pip install -r requirements.txt`

if the tests work correctly:

`cd tests/`

`nosetests`

you're good to go.

## API Reference

Implemented agents at the moment are:
`RandomAgent`, `MENACEAgent`, `QLearningAgent`, `DeepQLearningAgent`.
Agents can be generated just by initializing the corresponding object, and all share the same basic interface.
`agent.get_move(game_state)` gets a game state and outputs the agent's chosen move. `agent.endgame(reward)` gets a reward, assigns it to the last move performed, and saves it in memory. Most agents perform learning with this call, except for those based on neural networks, which have separate methods for learning.

`Match` is a class representing a match between two agents, and gets initialized with `Match(player1, player2)`, where `player1` and `player2` are two agents. `match.play()` makes the two agents play the game until it's over, assigns rewards to the agents, and returns the result of the game. 

## Contributors

Just me.

## License

vanilla MIT license. Check out [LICENSE](LICENSE).
