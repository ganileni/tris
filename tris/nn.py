# Q-network learning agent is implemented here so that this is the only file
# that contains tensorflow as a dependency.

import tensorflow as tf
import numpy as np
from tris.agents import BaseAgent
from tris.functions import state_from_hash, softmax
from tris.rules import GameState


def make_fully_connected_layer(input_layer,
                               layer_size,
                               activation=tf.nn.relu,
                               layer_name='',
                               logs=False):
    """Returns a fully connected layer in tensorflow.

    Inputs:
        input_layer: A tensor that serves as input to the layer.
        layer_size: an integer determining the number of units in the layer.
        activation: a tensorflow activation funcion.
        layer_name: the name of the layer. This string is used to name
            layer, weight, and bias. If name == '', a random string
            is assigned as name.

    Outputs:
        (a, w, b) = (layer, weights, biases) tensors."""
    if not layer_name:
        layer_name = ''.join(str(x) for x in np.random.randint(9, size=10))  # assign random name
    w = tf.Variable(tf.truncated_normal([int(input_layer.shape[1]), layer_size]), name='w_' + layer_name)
    if logs: tf.summary.histogram('weights', w)
    b = tf.Variable(tf.truncated_normal([1, layer_size]), name='b_' + layer_name)
    if logs: tf.summary.histogram('biases', b)
    z = tf.add(tf.matmul(input_layer, w), b, name='z_' + layer_name)
    if logs: tf.summary.histogram('pre-activations', z)
    a = activation(z, name='a_' + layer_name)
    if logs: tf.summary.histogram('activations', a)
    return a, w, b


def make_fully_connected_network(input_layer, architecture, activation=tf.nn.relu, network_name=''):
    """Returns a fully connected network.

    Inputs:
        input_layer: a tensor that will serve as input to the network
        architecture: a list of integers, containing the size of each hidden layer
            the size of the output layer.
        activation: a tensorflow activation funcion.
        network_name: the name of the network. This string is used to name all layers,
            weights, and biases.

    Outputs:
        (L, W, B) == (layers, wieghts, biases) lists of tensors."""
    if not architecture: raise AssertionError('no hidden layers in the architecture')
    L = [input_layer]
    W = []
    B = []
    for l, layer_size in enumerate(architecture):
        a, w, b = make_fully_connected_layer(
            L[-1],
            layer_size,
            activation=activation,
            layer_name=network_name + '_layer_' + str(l + 1)
        )
        L.append(a)
        W.append(w)
        B.append(b)
    return L, W, B


# this class will feed the data to the network
class DataFeeder():
    """A simple class that generates random batches from a dataset.

    DataFeeder(data : list):
        Initializes a feed_data object; data is a list.

    feed_data.next_batch(size : int):
        Returns a batch of length `size` by random sampling from the dataset.
        The result is a list of numpy arrays with the same order as the
        data list that was used for initialization."""

    def __init__(self, data: list, batch_size=100, verbose=False):
        self.data = data
        self.dataset_size = sizes[0]
        self.batch_size = int(batch_size)
        self.order = []
        self.epoch = -1
        self.verbose = verbose

    def next_batch(self):
        # if self.order contains less elements than a batch
        while len(self.order) <= self.batch_size:
            # generate new random indices
            new_order = list(range(self.dataset_size))
            np.random.shuffle(new_order)
            # add them to self.order
            self.order += new_order
            # note that you now have one more epoch
            self.epoch += 1
            if self.verbose: print('new epoch: ', self.epoch)
        # if enough random indices in self.order, select the first `batch.size` elements
        index, self.order = self.order[:self.batch_size], self.order[self.batch_size:]
        # and return the corresponding elements of the data
        return [x[index] for x in self.data]

    def next_batch_random(self, size):
        """just pick 'size' elements at random"""
        index = np.random.randint(0, self.dataset_size, size=int(size))
        return ([x[index, :] for x in self.data])


class DeepQLearningAgent(BaseAgent):
    """Implements a deep-Q learning agent, with a Q-network for approximating
    the Q function and softmax policy choice.

    The Q-network is a fully connected ANN with arbitrary architecture.

    It's an overkill for tic-tac-toe, but the point is to exercise."""

    def __init__(self, temperature=1, learning_rate=.1, discount=.9,
                 penalty=.01, architecture=[9, 9], activation=tf.nn.relu,
                 penalty_function=tf.nn.l2_loss,
                 optimizer_algo=tf.train.RMSPropOptimizer, optimizer_params=dict()):
        super().__init__()
        # boltzmann distribution temperature
        self.temperature = temperature
        # time discount for future rewards
        self.discount = discount
        # for regularization
        self.penalty = penalty
        self.penalty_function = penalty_function
        # optimizer
        self.optimizer_algo = optimizer_algo
        self.optimizer_parameters = optimizer_params
        self.learning_rate = learning_rate  # ANN learning rate
        # last layer of ANN is one single float, Q(s,a), so add [1]
        self.architecture = architecture + [1]
        self.activation = activation
        self._make_graph()
        self._start_session()

    def decide_move(self, game_state: float):
        # get GameState object from hash
        game_state = self.state_space[game_state]
        # evaluate Q-values with ANN
        Q_values = self._predict(game_state)
        # softmax choice
        choice = np.random.choice(list(game_state.actions.keys()),
                                  size=1,
                                  p=softmax(Q_values, self.temperature))[0]
        return game_state.actions[choice].coordinates, choice

    def endgame(self, result):
        # just save the states, the training will be done elsewhere
        pass

    def _make_graph(self):
        # this resets the whole default graph for tensorflow
        tf.reset_default_graph()
        # inputs/outputs:
        # each input example will be two np.hstacked 3x3 matrices, flattened
        # (initial state s and final state s' after selecting action a)
        self.input = tf.placeholder(tf.float32, [None, 3 * 6])
        self.layers, self.weights, self.biases = \
            make_fully_connected_network(
                input_layer=self.input,
                architecture=self.architecture,
                activation=self.activation
            )
        self.output = self.layers[-1]
        self.observed = tf.placeholder(tf.float32, shape=[None, 1])
        # MSE loss function
        self.loss = tf.reduce_sum(tf.square(self.output - self.observed))
        if self.penalty:
            penalty_tensor = tf.add_n([self.penalty_function(x) for x in self.weights])
            self.loss = self.loss + self.penalty * penalty_tensor
        self.optimizer = (self.optimizer_algo(learning_rate=self.learning_rate, **self.optimizer_parameters)
                          .minimize(self.loss))

    def _start_session(self):
        # start the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _batch(self):
        pass

    def _stats(self):
        pass

    def train(self):
        pass

    def _predict(self, game_state: GameState):
        # extract state-action pairs
        actions = [state_from_hash(_) for _ in game_state.actions]
        pairs = np.array([np.hstack((game_state.state, _)).flatten() for _ in actions])
        values = self.sess.run(self.output, feed_dict={self.input: pairs})
        return values.flatten()
