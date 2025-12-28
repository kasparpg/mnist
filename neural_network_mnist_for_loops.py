# Use Python 3.8 or newer (https://www.python.org/downloads/)
import math
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random
import csv
from csv import reader
from mnist import MNIST
import pandas as pd

class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layers: int, output_dim: int) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 10

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 0.05

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 10

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim

        self.neural_network = {}
        self.weights = []

        # create a neural net in a dictionary. Format: {layer: {node value 1: [node 1 weights]}}

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'train-images-idx3-ubyte')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        mndata = MNIST(os.path.join(project_root, 'data', 'mnist'))

        self.x_train, self.y_train = mndata.load_training()
        self.x_test, self.y_test = mndata.load_training()

    def initialize_neural_network(self):
        # initialize an array for the 1st layer

        self.neural_network[1] = []
        lower_limit = -0.5
        upper_limit = 0.5

        # initialize 30 nodes for the 1st layer
        for i in range(self.input_dim):
            node = [0, []]
            self.neural_network[1].append(node)

        # if there is a hidden layer
        if self.hidden_layers > 0:

            # assign 25 weights for each of the 30 nodes in the 1st layer.
            for i in range(self.input_dim):
                for j in range(self.hidden_units):
                    # each weight is a float value between -0.5 and 0.5
                    weight = round(random.uniform(lower_limit, upper_limit), 2)
                    self.neural_network[1][i][1].append(weight)

            for l in range(2, self.hidden_layers+2):

                # initialize an array for the l layer
                self.neural_network[l] = []
                if l == self.hidden_layers+1:
                    for i in range(self.hidden_units):
                        node = [0, []]
                        self.neural_network[l].append(node)
                        for j in range(self.output_dim):
                            # each weight is a float value between -0.5 and 0.5
                            weight = round(random.uniform(lower_limit, upper_limit), 2)
                            self.neural_network[l][i][1].append(weight)
                    break



                # initialize 25 nodes for the 2nd layer.
                for i in range(self.hidden_units):
                    node = [0, []]
                    self.neural_network[l].append(node)
                    for j in range(self.hidden_units):
                        # each weight is a float value between -0.5 and 0.5
                        weight = round(random.uniform(lower_limit, upper_limit), 2)
                        self.neural_network[l][i][1].append(weight)


            # initialize output layer.
            self.neural_network[self.hidden_layers+2] = []
            for i in range(self.output_dim):
                self.neural_network[self.hidden_layers+2].append([0, []])
        else:

            # initialize a weight for each node in the 1st layer.
            for i in range(self.input_dim):

                # each weight is a float value between -0.5 and 0.5
                weight = round(random.uniform(upper_limit, lower_limit), 2)
                self.neural_network[1][i][1].append(weight)

            # initialize 2nd layer. Since there is only 1 output node, it is assigned a temporary value of 0.
            self.neural_network[2] = []
            self.neural_network[2].append([0, []])

    def train(self) -> None:

        # initalize a neural network.
        self.initialize_neural_network()

        # Loop through all the examples.
        for e in range(self.epochs):
            for x, y in zip(self.x_train, self.y_train):


                In = {}

                #initialize a
                a = self.initialize_a_with_inputs(x)

                # get the number of layers.
                L = len(list(self.neural_network.keys()))

                # forward pass each layer, and calculate a and in for each node in layer l
                for l in range(2, L+1):
                    a[l], In[l] = self.forward_pass(l, a, In)

                # calculate delta(s) for the output node(s) in layer L
                delta = {}
                delta[L] = []
                for j in range(len(self.neural_network[L])):
                    correct_y = 0
                    if y == j+1:
                        correct_y = 1

                    delta[L].append(sigmoid_derivative(In[L][j])*(correct_y-a[L][j]))

                # calculate deltas for the other layers.
                for l in range(L-1, 0, -1):

                    # if we reach the input layer, we can break the loop.
                    if l == 1:
                        break

                    delta[l] = []
                    # for each node i in layer l
                    for i in range(len(self.neural_network[l])):

                        # g'(in_i). The in value for node i.
                        g_derivative_in_i = sigmoid_derivative(In[l][i])

                        # weight from node i in layer l to node j in layer l+1
                        w_i_j = 0

                        # for each weight j in node i
                        for j in range(len(self.neural_network[l][i][1])):
                            w_i_j += self.neural_network[l][i][1][j]*delta[l+1][j]
                        delta[l].append(g_derivative_in_i*w_i_j)

                # update all weights for the neural network:
                for l in range(1, L+1):
                    for i in range(len(self.neural_network[l])):
                        for j in range(len(self.neural_network[l][i][1])):
                            # weight from node i in layer l to layer l+1
                            weight = self.neural_network[l][i][1][j]
                            weight_i_j = weight + self.lr*a[l][i]*delta[l+1][j]
                            self.neural_network[l][i][1][j] = weight_i_j

        print("Training completed.")


    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """

        # get the number of layers
        L = len(list(self.neural_network.keys()))

        # initialize a
        a = self.initialize_a_with_inputs(x)

        # initialize in
        In = {}

        for l in range(2, L+1):
            a[l], In[l] = self.forward_pass(l, a, In)
        return [a[L].index(max(a[L])), a[L]]

    def forward_pass(self, l, a, In):
        a[l] = []
        In[l] = []
        # loop through each node j in layer l
        for j in range(len(self.neural_network[l])):
            in_j = 0
            # find all weights from nodes i in layer the previous layer pointing to node j.
            for i in range(len(self.neural_network[l - 1])):
                # weight from node i in layer l-1 to node j in layer l
                weight_i_j = self.neural_network[l - 1][i][1][j]
                # a_i is the a value for node i in layer l-1
                a_i = a[l - 1][i]
                # in_j is all the weights from the previous layer multiplied with their corresponding a value, pointion to node j
                in_j += (weight_i_j * a_i)

            In[l].append(in_j)
            a[l].append(sigmoid(in_j))
        return a[l], In[l]

    def initialize_a_with_inputs(self, x):
        # initialize a
        a = {}

        # a for the 1st layer as an empty array
        a[1] = []

        # append each input node value to 1st layer of a
        for i in range(self.input_dim):
            a[1].append(x[i])
        return a

    def print_nn_pretty(self):
        for l in list(self.neural_network.keys()):
            print("\nLayer "+str(l)+":")
            for i in range(len(self.neural_network[l])):
                print("   Node "+str(i+1)+", weights:", self.neural_network[l][i][1])

    def print_nn_stats(self):
        for l in list(self.neural_network.keys()):
            print("\nLayer " + str(l) + ":")
            print(" - Nodes:", len(self.neural_network[l]))
            print("  -> Weights per node:", len(self.neural_network[l][0][1]))

    def save_neural_network(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        w = csv.writer(open(os.path.join(project_root, 'data', 'neural_network.csv'), "w"))
        for key, value in self.neural_network.items():
            w.writerow([key, value])
        print("Neural network export completed.")

    def load_neural_network(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(project_root, 'data', 'neural_network.csv'), 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                print(row)
                layer = int(row[0])
                data = list(np.array(row[1]))

                self.neural_network[layer] = data


def sigmoid_derivative(x):
    x = np.clip(x, -500, 500)
    return sigmoid(x)*(1-sigmoid(x))


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, 0, 1)
        accuracy = self.get_accuracy()
        print("Perceptron accuracy:", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, 1, 1)
        accuracy = self.get_accuracy()
        print("Hidden accuracy:", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    # unittest.main()
    """Input Dimension configuration """
    input_dim = 784
    output_dim = 10
    hidden_layers = 2

    """Neural Network Training"""
    nn = NeuralNetwork(input_dim, hidden_layers, output_dim)
    nn.load_data()

    nn.train()
    nn.save_neural_network()

    """ Neural Network Load """
    #nn.load_neural_network()
    #nn.print_nn_stats()

    """Neural Network Accuracy"""
    #number = random.randint(0,10000)
    #xd = nn.x_test[number]
    #prediction = nn.predict(xd)
    #print("\nPrediction: number was",nn.y_test[number], "prediction was", prediction[0],"| Details: ", prediction[1])


