#!/usr/bin/Python3
import math
from parameters import Parameters as p
import numpy as np

class NN:

    def __init__(self):
        self.input_bias = 1.0  # Biasing weight for input layer weights
        self.output_bias = 1.0   # Biasing weight for hidden layer weights
        self.n_inputs = p.ninputs
        self.num_in_weights = (self.n_inputs + 1) * p.hsize
        self.num_out_weights = p.noutputs * (p.hsize + 1)
        self.n_weights = self.num_in_weights + self.num_out_weights
        self.n_outputs = p.noutputs
        self.h_layer_size = p.hsize

        # Initialize vectors
        self.in_vec = np.zeros(self.n_inputs)  # Input Layer
        self.weight_vec = np.zeros(self.n_weights)
        self.out_vec = np.zeros(self.n_outputs)  # Output Layer
        self.h_layer = np.zeros(self.h_layer_size)  # Hidden Layer

    def create_NN(self, ninput, noutput, hsize):
        self.n_inputs = ninput
        self.num_in_weights = (self.n_inputs+1)*hsize
        self.num_out_weights = noutput*(hsize+1)
        self.n_weights = self.num_in_weights + self.num_out_weights
        self.n_outputs = noutput
        self.h_layer_size = hsize

        # Initialize vectors
        self.in_vec = np.zeros(self.n_inputs)  # Input Layer
        self.weight_vec = np.zeros(self.n_weights)
        self.out_vec = np.zeros(self.n_outputs)  # Output Layer
        self.h_layer = np.zeros(self.h_layer_size)  # Hidden Layer

    def get_inputs(self, state_vec):  # Input states are x distance to the goal and y distance to the goal
        i = 0
        while i < self.n_inputs:
            self.in_vec[i] = state_vec[i]
            i += 1

    def get_weights(self, evo_weights):  # Weights taken from GA population (passed into function)
        i = 0
        while i < self.n_weights:
            self.weight_vec[i] = evo_weights[i]
            i += 1

    def get_ouput(self):
        action = -1; count = 0
        self.h_layer = [0] * self.h_layer_size  # Hidden Layer resets to values of 0
        self.out_vec = [0] * self.n_outputs  # Output Layer resets to values of 0

        for i in range(self.n_inputs):  # Pass inputs to hidden layer
            for j in range(self.h_layer_size):
                self.h_layer[j] += self.in_vec[i]*self.weight_vec[count]
                count += 1
        for j in range(self.h_layer_size):  # Add Biasing Node
            self.h_layer[j] += (self.input_bias * self.weight_vec[count])
            count += 1

        for i in range(self.h_layer_size):  # Pass through sigmoid
            self.h_layer[i] = self.sigmoid(self.h_layer[i])

        for i in range(self.h_layer_size):  # Pass from hidden layer to output layer
            for j in range(self.n_outputs):
                self.out_vec[j] += self.h_layer[i]*self.weight_vec[count]
                count += 1
        for j in range(self.n_outputs):  # Add biasing node
            self.out_vec[j] += (self.output_bias * self.weight_vec[count])
            count += 1
        assert (count == self.n_weights)  # Count is meant to go through the entire range of weights

        for i in range(self.n_outputs):  # Pass through sigmoid
            self.out_vec[i] = self.sigmoid(self.out_vec[i])

        if self.out_vec[2] < 0.5:  # 3rd output determines if agent moves in X or Y direction
            if self.out_vec[0] < 0.5:  # Output 1 determines if agent moves in pos X or neg X
                action = 0
            else:
                action = 1

        if self.out_vec[2] >= 0.5:
            if self.out_vec[1] < 0.5:  # Output 2 determines if agent moves in pos Y or neg Y
                action = 2
            else:
                action = 3
        return action

    def sigmoid(self, inp):
        sig = 1/(1 + math.exp(-inp))
        return sig