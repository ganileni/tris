from tris.agents import MENACEAgent, QLearningAgent
from tris.nn import DeepQLearningAgent

SAVE_DIR = '../pickles/'
PICKLE_EXTENSION = '.pkl'
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


# TODO ------
Q_LEARNER_NAME = 'qlearner'
DEEPQ_LEARNER_NAME = 'deep_qlearner'
