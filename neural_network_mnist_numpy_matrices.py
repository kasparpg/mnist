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
import time


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
        self.hidden_units = 16

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 0.0001


        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.training = False
        self.epochs = 1000
        self.previous_epochs = 0

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.w1 = np.array([])
        self.w2 = np.array([])
        self.w3 = np.array([])
        # create a neural net in a dictionary. Format: {layer: {node value 1: [node 1 weights]}}

    def print_info(self):
        print("----------------------------")
        print("| MNIST Data set, digit AI |")
        print("----------------------------")
        print("- Learning Rate:", self.lr)
        print("- Hidden Layers:", self.hidden_layers, "("+str(self.hidden_layers+2)+" layers total)")
        print("- Hidden Layers Dimension:", self.hidden_units)
        print("- Input Dimensions:", self.input_dim)
        print("- Output Dimensions:", self.output_dim)


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
        print("\nStarted loading data...")
        # Get the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        mndata = MNIST(os.path.join(project_root, 'data', 'mnist'))

        self.x_train, self.y_train = mndata.load_training()
        self.x_test, self.y_test = mndata.load_testing()

        print("Data successfully loaded.")

    def f_forward(self, x, w1, w2, w3):

        z1 = x.dot(w1)
        a1 = sigmoid(z1)

        z2 = a1.dot(w2)
        a2 = sigmoid(z2)

        z3 = a2.dot(w3)
        a3 = sigmoid(z3)

        return a3

    def generate_wt(self, x, y):
        l = []

        for i in range(x * y):
            l.append(np.random.randn())
        return np.array(l).reshape(x, y)

    def loss(self, out, Y):
        s = (np.square(out - Y))
        s = np.sum(s) / len(Y)
        return s

    def back_prop(self, x, y, w1, w2, w3, alpha):

        # w1.shape = 784, 16
        # w2.shape = 16, 16
        # w3.shape = 16, 10

        """ in and a calculations for all 4 layers. """
        # Layer 1 (input layer)
        a1 = x  # shape=(1, 784)

        # Layer 2 (first hidden Layer)
        in2 = a1.dot(w1)  # shape=(1, 16)
        a2 = sigmoid(in2)  # shape=(1, 16)

        # Layer 3 (second hidden Layer)
        in3 = a2.dot(w2)  # shape=(1, 16)
        a3 = sigmoid(in3)  # shape=(1, 16)

        # Layer 4 (output layer)
        in4 = a3.dot(w3)  # shape=(1, 10)
        a4 = sigmoid(in4)  # shape=(1, 10)

        """ delta calculations for layers 2, 3, 4."""
        # Delta for layer 4 (output layer)
        d4 = sigmoid_derivative(in4)*(y - a4)  # shape=(1, 10)

        # Delta for layer 3 (second hidden layer)
        d3 = sigmoid_derivative(in3) * ((w3.dot(d4.transpose())).transpose())  # shape=(1, 16)

        # Delta for layer 2 (first hidden layer)
        d2 = sigmoid_derivative(in2) * ((w2.dot(d3.transpose())).transpose())  # shape=(1, 16)

        """ updating the weights. """
        # Weight update for weights from layer 1 to layer 2
        w1 = w1 + (alpha * (a1.transpose()).dot(d2))

        # Weight update for weights from layer 2 to layer 3
        w2 = w2 + (alpha * (a2.transpose()).dot(d3))

        # Weight update for weights from layer 3 to layer 4
        w3 = w3 + (alpha * (a3.transpose()).dot(d4))

        return w1, w2, w3

    def train(self):
        print("\nTraining started...")
        self.training = True

        if self.previous_epochs == 0:
            self.w1 = self.generate_wt(self.input_dim, self.hidden_units)  # 784, 16
            self.w2 = self.generate_wt(self.hidden_units, self.hidden_units)  # 16, 16
            self.w3 = self.generate_wt(self.hidden_units, self.output_dim)  # 16, 10
        else:
            print("Weights were not reset due to previous epochs:", self.previous_epochs)
        print("Number of epochs for this training session:", self.epochs)
        print("Training examples per epoch:", len(self.x_train))
        epoch_times = []
        for e in range(self.epochs):
            start_time = time.time()
            count = 1
            examples_length = len(self.x_train)
            print(" -> Currently at epoch", e + 1, "of", self.epochs,
                  "epoch(s)... 0%", end='')
            for example_x, example_y in zip(self.x_train, self.y_train):
                x = np.array(example_x).reshape(1, self.input_dim)
                y = np.zeros(output_dim, dtype=int)
                y[example_y] = 1
                self.w1, self.w2, self.w3 = self.back_prop(x, y, self.w1, self.w2, self.w3, self.lr)
                if count % 500 == 0:
                    print("\r" + " -> Currently at epoch", e + 1, "of", self.epochs,
                          "epoch(s)... " + str(round((count / examples_length) * 100, 2)) + "% ", end='')
                count += 1

            epoch_times.append(round(time.time() - start_time, 2))
            time_remaining = str(
                math.floor(((sum(epoch_times) / (e + 1)) * (self.epochs - (e + 1))) / 60)) + " min " + str(
                int((sum(epoch_times) / (e + 1)) * (self.epochs - (e + 1)) % 60)) + "s."
            print("Epoch", e + 1, "complete (" + str(epoch_times[e]) + "s).",
                  "Estimated time remaining: " + time_remaining)
            self.accuracy()

        print("Training finished.")

    def predict(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        print("\nPrediction started...")
        print("Prediction involves a random example.")
        out = self.f_forward(x, self.w1, self.w2, self.w3)
        prediction = np.where(out == max(out))[0][0]
        print(" -> Number was:", y)
        print(" -> Prediction was:", prediction)
        print(" -> Details:", out)
        print("Prediction finished.")

        return prediction

    def accuracy(self):
        # print("\nAccuracy test started...")
        correct = 0
        test_length = len(self.y_test)
        for test_x, test_y in zip(self.x_test, self.y_test):
            x = np.array(test_x).reshape(1, self.input_dim)
            y = np.array(test_y)
            out = self.f_forward(x, self.w1, self.w2, self.w3)[0]
            prediction = np.where(out == max(out))
            if prediction == y:
                correct += 1
        print("   -> Accuracy of " + str(test_length) + " examples:", round(((correct / test_length) * 100), 2), "%.")
        # print("Accuracy test finished.")
        return correct / test_length

    def save_weights(self):
        print("\nStarted saving weights...")
        total_epochs = self.previous_epochs
        if self.training:
            total_epochs += self.epochs
        weights = np.array([self.w1, self.w2, self.w3, total_epochs], dtype="object")
        project_root = os.path.dirname(os.path.abspath(__file__))
        np.save(os.path.join(project_root, 'weights', 'weights_'+str(self.lr)), weights)


        print("Weights successfully saved.", total_epochs, "total epochs.")


    def load_weights(self):
        print("\nStarted loading weights...")
        project_root = os.path.dirname(os.path.abspath(__file__))
        weights = np.load(os.path.join(project_root, 'weights', 'weights_'+str(self.lr)+'.npy'), allow_pickle=True)
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.w3 = weights[2]
        self.previous_epochs = weights[3]
        print("Weights (weights_" + str(self.lr) + ".npy) successfully loaded.", self.previous_epochs, "total epochs.")

def sigmoid_derivative(x):
    x = np.clip(x, -500, 500)
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # unittest.main()

    """Input Dimension configuration """
    input_dim = 784
    output_dim = 10

    """Neural Network Training"""
    nn = NeuralNetwork(input_dim, 2, output_dim)
    nn.print_info()
    nn.load_data()

    nn.load_weights()
    nn.train()
    nn.save_weights()



    # number = random.randint(0, 10000)
    # example_x = np.array(nn.x_test[number])
    # example_y = np.array(nn.y_test[number])
    # nn.predict(example_x, example_y)


