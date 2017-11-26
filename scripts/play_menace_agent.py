from tqdm import tqdm

from tris.rules import Match
from tris.agents import RandomAgent, MENACEAgent

# play a game between two random agents, print the moves and the result

# initialize agents and match
beads_n = 100
menace, player2 = MENACEAgent(beads_n=beads_n), RandomAgent()
match = Match(menace, player2)
# play loop until end game
match.play()
for state in menace.state_space:
    state_actions = menace.state_space[state].actions
    for action in state_actions:
        if state_actions[action].value != beads_n:
            print(state, action, state_actions[action].value)

# now  play 100 games
match_results = []
for _ in tqdm(range(10000)):
    match = Match(menace, player2)
    match_results.append(match.play())
all_actions = 0
changed_states = 0
for state in menace.state_space:
    state_actions = menace.state_space[state].actions
    all_actions += len(state_actions.keys())
    for action in state_actions:
        if state_actions[action].value != beads_n:
            changed_states += 1
print(changed_states / all_actions)
