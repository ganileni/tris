# Q-network learning agent is implemented here so that this is the only file
# that contains tensorflow as a dependency.

import tensorflow as tf
import numpy as np
from tris.agents import BaseAgent
from tris.functions import state_from_hash, softmax, pickle_save, pickle_load
from tris.rules import GameState
from copy import copy

CHECKPOINT_EXTENSION = '.ckpt'


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


class DataFeeder():
    """A simple class that generates random batches from a dataset.

    DataFeeder(data : list, batch_size : int):
        Initializes a feed_data object; data is a list of ndarrays. each of these arrays
        will be split in batches and fet as output, i.e. if `data` = [X, Y, ...],
        then self.next_batch() will return [batch_X, batch_Y, ...].

    feed_data.next_batch():
        Returns a batch of length `size` by random sampling from the dataset.
        data is shuffled automatically before being split into batches."""

    def __init__(self, data: list, batch_size=100, verbose=False):
        self.data = data
        self.dataset_size = len(data[0])
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
        """just pick `size` elements at random"""
        index = np.random.randint(0, self.dataset_size, size=int(size))
        return ([x[index, :] for x in self.data])


class BiasedDataFeeder(DataFeeder):
    """Generates batches from the dataset sampling with an
    arbitrary distribution."""

    def set_distribution(self, distribution):
        self.distribution = distribution

    def next_batch(self):
        index = np.random.choice(list(range(self.dataset_size)),
                                 size=self.batch_size,
                                 replace=True,
                                 p=self.distribution)
        return ([x[index, :] for x in self.data])


class DeepQLearningAgent(BaseAgent):
    """Implements a deep-Q learning agent, with a Q-network for approximating
    the Q function and softmax policy choice.
    The Q-network is a fully connected ANN with arbitrary architecture.
    All the initialization parameters default to the best known parameters for tic-tac-toe.3

    (It's an overkill for tic-tac-toe, but the point is to exercise.)

    Args:
        temperature: temperature of the softmax distribution used to choose next state from the Q-values of alternative possibilities.
        learning_rate: by how much to change the Q-values at each learning iteration.
        discount: by how much to discount future rewards while training.
        policy: if 'softmax' (default behavior) the policy picks actions using a softmax distribution (and the epsilon parameter is ignored). if 'epsilon', epsilon-greedy policy (the temperature parameter is ignored)
        epsilon: the epsilon parameter for the epsilon-greedy policy: pick a random action epsilon % of the times (between 0 and 1)
        penalty_function: what kind of penalty for the parameters of the neural network, defaults to L2 loss.
        penalty: this coefficient multiplies the penalty.
        architecture: a list of integers describing the ANN. each element of the list corresponds to the number of neurons in subsequent layers of the fully connected ANN approximating the Q-function. output layer is automatically added by the init function.
        activation: activation function of the neurons in the ANN.
        optimizer_algo: what algorithm to use during training of the ANN. defaults to RMSProp.
        optimizer_params: dict of parameters to pass to the optimizer. defaults to empty dict.
        batch_size: batch size to train the ANN.
        epochs: how many epochs to use in training.
        """

    def __init__(self, temperature=1, learning_rate=.1, discount=.9,
                 policy='softmax', epsilon=0.1,
                 penalty=.01, architecture=[9, 9], activation=tf.nn.sigmoid,
                 penalty_function=tf.nn.l2_loss, epochs=20,
                 optimizer_algo=tf.train.RMSPropOptimizer, optimizer_params=dict(),
                 batch_size=500):
        super().__init__()
        # this dict contains the parameters necessary to save the agent to disk
        self.save_attrs = dict()
        # boltzmann distribution temperature
        self.save_attrs['temperature'] = temperature
        # time discount for future rewards
        self.save_attrs['discount'] = discount
        # for ANN regularization
        self.save_attrs['penalty'] = penalty
        self.save_attrs['penalty_function'] = penalty_function
        # ANN SGD optimizer
        self.save_attrs['optimizer_algo'] = optimizer_algo
        self.save_attrs['optimizer_params'] = optimizer_params
        self.save_attrs['batch_size'] = batch_size
        self.save_attrs['epochs'] = epochs
        # ANN learning rate
        self.save_attrs['learning_rate'] = learning_rate
        self.save_attrs['architecture'] = architecture
        self.save_attrs['activation'] = activation
        self.save_attrs['epsilon'] = epsilon
        self.save_attrs['policy '] = policy
        if policy == 'softmax':
            self.decision = self._decision_softmax
        elif policy == 'epsilon':
            self.decision = self._decision_epsilon
        # put all the attributes in the agent object
        for key in self.save_attrs:
            if key == 'architecture':
                # last layer of ANN is one single float, Q(s,a),
                # so we need to add [1] to the architecture
                setattr(self, key, self.save_attrs[key] + [1])
            else:
                setattr(self, key, self.save_attrs[key])
        # generate ANN graph in tensorflow
        self._make_graph()
        self._start_session()
        # memory of past games goes here: each game is a list of Steps
        self.examples = []

    def decide_move(self, game_state: float):
        # get GameState object from hash
        game_state = self.state_space[game_state]
        # evaluate Q-values with ANN
        Q_values = self._predict(game_state)
        # softmax choice
        choice = self._decision(Q_values, game_state)
        return game_state.actions[choice].coordinates, choice

    def _decision_softmax(self, Q_values, game_state):
        choice = np.random.choice(list(game_state.actions.keys()),
                                  size=1,
                                  p=softmax(Q_values, self.temperature))[0]
        return choice

    def _decision_epsilon(self, Q_values, game_state):
        # epsilon % of the times random
        if np.random.uniform() < self.epsilon:
            choice = np.random.choice(list(game_state.actions.keys()),
                                      size=1,
                                      )[0]
        else:
            choice = list(game_state.actions.keys())[np.argmax(Q_values)]
        return choice

    def endgame(self, reward):
        # just save the observed states and rewards for the episode,
        #  the training will be done elsewhere
        self.examples.append((copy(self.history), reward))
        self.history = []

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
        self.optimizer = (self.optimizer_algo(learning_rate=self.learning_rate, **self.optimizer_params)
            .minimize(self.loss))

    def _start_session(self):
        # start the session
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _batch(self):
        # y for loss will be:
        # y = reward + discount* max_{a'}[Q(s',a')] := reward + q_sa
        # state_action is the (s,a) pair to be shown as input to the ANN
        state_action, q_sa, reward = self.dataset_feeder.next_batch()
        # these two assignments to variable target are just for reshaping
        # from (batch_size,) to (batch_size,1)
        # TODO: rewrite in a more efficient way
        target = q_sa + reward
        target = np.array([[_] for _ in target])
        self.sess.run(self.optimizer, feed_dict={self.input: state_action,
                                                 self.observed: target,
                                                 })

    def _train_stats(self, iteration_number):
        print(iteration_number / self.iterations)
        pass

    def learn(self, purge_memory=True):
        observed_inputs, observed_reward, predicted_outputs, distance_from_reward = self._preprocess_experience()
        # now train. DataFeeder automatically reshuffles data.
        self.dataset_feeder = DataFeeder(
                [observed_inputs, predicted_outputs, observed_reward],
                batch_size=self.batch_size)
        # determine number of iterations:
        self.iterations = int(self.epochs * len(observed_inputs) / self.batch_size)
        for _ in range(self.iterations):
            self._batch()
            # TODO: write a function that computes and prints training stats
            # if _ % 1000:
            #     self._train_stats(_)
        if purge_memory:
            self.purge_memory()

    def _preprocess_experience(self):
        """processes the experience of the agent which is found in self.examples
        so that it can be fed to the Q function approximator
        returns:
            observed_inputs: the observed state and action taken
            observed_reward: reward immediately observed after the action
            predicted_output: the Q-value as predicted by the approximator
            distance_from_reward : how many timesteps before the reward
             """
        observed_inputs = []
        observed_reward = []
        predicted_outputs = []
        distance_from_reward = []
        next_state = []
        # process inputs and outputs to train the net
        for episode in self.examples:
            episode_match, example_reward = episode
            last_step = True
            for n, step in enumerate(reversed(episode_match)):
                this_state = state_from_hash(step.state_t)
                next_state.append(state_from_hash(step.action_t))
                observed_inputs.append(np.hstack((this_state,
                                                  this_state != next_state[-1]))
                                       .flatten())
                distance_from_reward.append(n)
                # now we have to evaluate max_{s'}[Q(a',s')]
                # let's see all possible actions two steps ahead
                two_ahead = []
                for possible_action in self.state_space[step.action_t].actions:
                    possible_action = state_from_hash(possible_action)
                    two_ahead.append(np.hstack((next_state[-1],
                                                next_state[-1] != possible_action))
                                     .flatten())
                if not two_ahead:
                    # if it's a terminal state, no two-ahead, so set the max to 0
                    max_next_state = 0
                else:
                    # evaluate Q on the two-ahead actions
                    two_ahead = np.array(two_ahead)
                    two_ahead[two_ahead == 2] = -1
                    max_next_state = self.sess.run(
                            self.output,
                            feed_dict={self.input: two_ahead}).flatten()

                    # calc the maximum
                    max_next_state = np.max(max_next_state)
                predicted_outputs.append(max_next_state)
                if last_step:
                    # because we start from last step, `last_step` will be true
                    observed_reward.append(example_reward)
                    # then set it to false so non-last steps get reward 0
                    last_step = False
                else:
                    observed_reward.append(0)
        # Q-network output from the inputs
        predicted_outputs = self.discount * np.vstack(predicted_outputs).flatten()
        observed_inputs = np.array(observed_inputs)
        # possible max value in a state is 2, set all 2's to -1's
        observed_inputs[observed_inputs == 2] = -1
        observed_reward = np.vstack(observed_reward).flatten()
        return observed_inputs, observed_reward, predicted_outputs, distance_from_reward

    def purge_memory(self):
        self.examples = []

    def _predict(self, game_state: GameState):
        # extract state-action pairs
        actions = [state_from_hash(_) != game_state.state for _ in game_state.actions]
        pairs = np.array([np.hstack((game_state.state, _)).flatten() for _ in actions])
        pairs[pairs == 2] = -1
        values = self.sess.run(self.output, feed_dict={self.input: pairs})
        return values.flatten()

    def save_agent(self, path):
        """save functionality is different in case of agents that contain ANN's"""
        # save all parameters needed to reconstruct the agent
        pickle_save(self.save_attrs, path)
        # initialize tensorflow saver
        saver = tf.train.Saver(var_list=self._variables_to_save())
        saver.save(self.sess, path + CHECKPOINT_EXTENSION)

    def _variables_to_save(self):
        """generate a list of ANN variables to be saved to disk (in dict form, tensorflow-readable)"""
        save_vars = self.weights + self.biases
        save_names = [_.name for _ in save_vars]
        save_dict = {name: var for name, var in zip(save_names, save_vars)}
        return save_dict

    @classmethod
    def load_agent(cls, path):
        save_attrs = pickle_load(path)
        agent = cls(**save_attrs)
        saver = tf.train.Saver(var_list=agent._variables_to_save())
        saver.restore(agent.sess, path + CHECKPOINT_EXTENSION)
        return agent
