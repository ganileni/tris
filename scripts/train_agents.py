from tris.agents import RandomAgent, MENACEAgent
from tris.nn import DeepQLearningAgent
from tris.rules import Match
from tqdm import tqdm
from os import mkdir
from os.path import dirname
import numpy as np


def create_dir(path):
    """just a small routine that makes a directory if it doesn't exist"""
    try:
        mkdir(dirname(path))
    except FileExistsError:
        pass


def train_Random(random_description):
    """train proedure for RandomAgent."""
    pass


def train_MENACE(menace_description):
    """train proedure for MENACEAgent."""
    menace = MENACEAgent()
    player2 = RandomAgent()
    match_results = []
    for _ in tqdm(range(int(1e5))):
        match = Match(menace, player2)
        match_results.append(match.play())
    create_dir(menace_description.path)
    menace.save_agent(menace_description.path)
    return match_results


def train_QLearning(qlearning_descritpion):
    """train proedure for QLearningAgent."""
    # TODO -- look into optimal exploration/exploitation balance
    learning_rate = 0.1
    discount = 0.9
    start_epsilon = 1
    convergence_epsilon = .1
    # initialize agents & data structures
    random_agent = RandomAgent()
    qlearner = QLearningAgent(
            discount=discount,
            learning_rate=learning_rate,
            epsilon=start_epsilon,
            policy='epsilon')
    # train
    for _ in tqdm(range(10000)):
        match = Match(qlearner, random_agent)
        match.play()
    qlearner.epsilon = convergence_epsilon
    for _ in tqdm(range(60000)):
        match = Match(qlearner, random_agent)
        match.play()
    create_dir(qlearning_description.path)
    qlearner.save_agent(qlearning_description.path)


def train_DeepQLearning(deepq_learning_description):
    """train proedure for DeepQLearningAgent."""
    # TODO -- look into optimal exploration/exploitation balance
    nn_epochs = 40
    games = np.log10(2500)
    train_stages = 40
    batch_size = .5  # as a percentage of dataset size
    architecture_depth = 2
    architecture_width = 50
    learning_rate = np.log10(.05)
    start_temperature = np.log10(1)
    cooling_down_rate = np.log10(1.01)
    # format parameters
    architecture = [int(architecture_width)] * int(architecture_depth)
    # exponentiation is because we want to optimize in log space
    games = int(10 ** games)
    batch_size = int(batch_size * games * 10)  # (10 ~ avg game length)
    learning_rate = 10 ** learning_rate
    start_temperature = 10 ** start_temperature
    cooling_down_rate = 10 ** cooling_down_rate
    # initialize agents & data structures
    random_agent = RandomAgent()
    deepq_learner = DeepQLearningAgent(
            epochs=nn_epochs,
            batch_size=batch_size,
            architecture=architecture,
            learning_rate=learning_rate,
            temperature=start_temperature)
    all_results = []
    # train
    for _ in tqdm(range(train_stages)):
        batch_result = []
        for _ in (range(games)):
            match = Match(deepq_learner, random_agent)
            # play loop until end game
            batch_result.append(match.play())
        deepq_learner.learn()
        deepq_learner.temperature = deepq_learner.temperature / cooling_down_rate
        all_results.append(batch_result)
    create_dir(deepq_learning_description.path)
    deepq_learner.save_agent(deepq_learning_description.path)
    return all_results
