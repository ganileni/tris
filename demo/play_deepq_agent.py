# from tris.rules import Match
# from tris.agents import RandomAgent
# from tris.nn import DeepQLearningAgent
# from tqdm import tqdm
#
# # play a game between a deep-Q agent and a random agent
#
# # initialize agents and match
# deepq_learner = DeepQLearningAgent(iterations=10)
# random_agent = RandomAgent()
# for _ in tqdm(range(100)):
#     match = Match(deepq_learner, random_agent)
#     # play loop until end game
#     match.play()
# deepq_learner.learn()



from tris.rules import Match
from tris.agents import RandomAgent
from tris.nn import DeepQLearningAgent
from tqdm import tqdm
from tris.functions import time_average
import numpy as np
import matplotlib
matplotlib.use('nbagg')
from matplotlib import pyplot as plt

# play a game between a deep-Q agent and a random agent

nn_epochs = 20
games = 50 #250
train_stages = 20#20*100
# initialize agents and match
deepq_learner = DeepQLearningAgent(
    iterations=500,
    batch_size = 100,
    architecture = [50,50])
random_agent = RandomAgent()

all_results = []
for _ in tqdm(range(train_stages)):
    batch_result = []
    #lookup how to make nested tqdms
    for _ in (range(games)):
        match = Match(deepq_learner, random_agent)
        # play loop until end game
        batch_result.append(match.play())
    examples = len(deepq_learner.examples)
    print(examples)
    deepq_learner.iterations = int(nn_epochs * examples/deepq_learner.batch_size)
    deepq_learner.learn()
    all_results.append(batch_result)