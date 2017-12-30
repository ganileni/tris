"""these are settings for agents used by the human_play.py script and training procedures found in train_agents.py.
The settings for each agent are stored in a class called AgentDescription."""

from scripts.constants import SAVE_DIR, PICKLE_EXTENSION
from scripts.train_agents import train_MENACE, train_QLearning, train_Random, train_DeepQLearning
from tris.agents import MENACEAgent, QLearningAgent, RandomAgent
from tris.nn import DeepQLearningAgent


class AgentDescription():
    save_dir = SAVE_DIR
    pickle_ext = PICKLE_EXTENSION
    def __init__(self,
                 name,
                 filename,
                 klass,
                 train_procedure,
                 ):
        self.name = name
        self.filename = filename
        self.klass = klass
        self.train = train_procedure
        self.path = self.save_dir + self.filename + self.pickle_ext


random_description = AgentDescription(
        name = 'Random',
        filename = 'none',
        klass = RandomAgent,
        train_procedure = train_Random
)
menace_description = AgentDescription(
        name = 'MENACE',
        filename='menace',
        klass = MENACEAgent,
        train_procedure = train_MENACE
)

qlearning_description = AgentDescription(
        name = 'Q-Learning',
        filename='qlearning',
        klass = QLearningAgent,
        train_procedure = train_QLearning
)

deepq_learning_description = AgentDescription(
        name = 'Deep-Q Learning',
        filename='deepq_qlearning',
        klass = DeepQLearningAgent,
        train_procedure = train_DeepQLearning
)