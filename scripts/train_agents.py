from tris.agents import RandomAgent, MENACEAgent
from tris.rules import Match
from tqdm import tqdm
from os import mkdir
from os.path import dirname

def create_dir(path):
    mkdir(dirname(path))


def train_MENACE(menace_description):
    menace = MENACEAgent()
    player2 = RandomAgent()
    match_results = []
    for _ in tqdm(range(int(5e5))):
        match = Match(menace, player2)
        match_results.append(match.play())
    create_dir(menace_description.path)
    menace.save_agent(menace_description.path)
    return match_results

def train_QLearning(qlearning_descritpion):
    pass