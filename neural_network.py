#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Union, List


# f logistyczna jako przyklad sigmoidalej
def sigmoid(x):
    size = x.size
    ones = np.ones(size)
    return ones/(ones+np.exp(-x))


#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    size = x.size
    ones = np.ones(size)
    s = ones/(ones+np.exp(-x))
    return s * (ones - s)


#f. straty
def nloss(y_out, y):
    (y_out - y) ** 2


#pochodna f. straty
def d_nloss(y_out, y):
    return 2*(y_out - y)


###############################################################################

class Neuron:
    def __init__(
        self, size: int, activation_function=sigmoid,
        wages_scale_factor=None, wage_value: float = None,
    ):
        self.size = size
        self.activation_funcion = activation_function
        if not wages_scale_factor:
            self.wages_scale_factor = 1/np.sqrt(self.size)
        # self.wages = wages # niech len(wages) = len(x) + 1 # because there is a bias at start for activation function
        self.activation_function = activation_function  # for example sigmoid
        self.initiate_wages(wage_value)  # they include BIAS

    def initiate_wages(self, wage_value=None):
        if wage_value == None:
            self.wages = self.wages_scale_factor * np.random.uniform(
                -self.size, self.size, self.size+1
            )  # last one is bias
            self.wages[-1] = 1
        elif wage_value == 0:
            self.wages = wage_value * np.ones(self.size+1)

    def evaluate(
        self, input: np.array, activation_function_included=True
    ) -> float:
        # TODO: Error probably involves the bias of a neuron (the last of wages - I was not consistant whith its usage and operations on it)
        print("In evaluate of Neuron")
        print(f'input:{input}')
        print(f'activation_function_included {activation_function_included}')
        print(f'self.wages: {self.wages}')
        result = (input * self.wages[:-1]).sum() + self.wages[-1]
        print(f"result: {result}")
        print("End")
        if activation_function_included:
            return self.activation_function(result)[0]
        return result


###############################################################################
class Layer:
    def __init__(
        self, input_size, neurons_num,
        activation_function=None, input_wage=None
    ):
        self.input_size = input_size  # without BIAS
        self.neurons_num = neurons_num
        self.activation_function = activation_function
        self.neurons = []
        self.generate_neurons(input_wage)

    def get_all_wages(self):
        return list(map(
            lambda neuron: neuron.wages, self.neurons
        ))

    def generate_neurons(self, wage=None):
        for i in range(self.neurons_num):
            self.neurons.append(
                Neuron(
                    self.input_size,
                    activation_function=self.activation_function,
                    wage_value=wage
                )
            )

    def propagate_backward_dsum(
        self, next_results: List[np.array], next_layer: Layer
    ) -> List[np.array]:
        # actual result_expected if next_layer != None
        # if next layer == None, next_result
        current_results = []
        for indx, neuron in enumerate(self.neurons):
            msum = np.add.reduce(np.array(list(map(
                lambda neuron_result, next_neuron:
                neuron_result*next_neuron.wages[indx],
                next_results, next_layer.neurons
            ))))
            current_results.append(d_sigmoid(msum))
        # print(current_results)
        return current_results

    def evaluate(
        self, input: Union[np.array, float], activation_function_included=True
    ) -> np.array:
        # if self.input_size == 1:
        #     input = np.array(input)
        return np.array(list(map(
            lambda neuron: neuron.evaluate(input, activation_function_included), self.neurons
        )))


###############################################################################
class DlNet:
    def __init__(
        self, x, y, number_of_layers, input_dimentionality=1,
        HIDDEN_L_SIZE=9, activation_function=sigmoid
    ):
        self.x = x
        self.y = y
        self.y_out = 0
        self.input_dimentionality = input_dimentionality
        self.HIDDEN_L_SIZE = HIDDEN_L_SIZE  # number of neurons in a single layer
        self.LR = 0.003  # learning rate
        self.input_layer = Layer(
            input_dimentionality, HIDDEN_L_SIZE, sigmoid
        )
        self.output_layer = Layer(self.HIDDEN_L_SIZE, 1, None, 0)
        # print(self.get_layer_neurons(self.output_layer)[0].wages)
        self.hidden_layers = []
        if number_of_layers > 2:
            for i in range(number_of_layers-2):
                self.hidden_layers.append(Layer(
                    self.HIDDEN_L_SIZE, self.HIDDEN_L_SIZE, activation_function
                ))

    def forward(self, x):
        every_layer_input = []
        current_input = x
        every_layer_input.append(np.concatenate([current_input, np.array([1])]))
        # every_layer_input.append(np.array(current_input+[1]))
        print(every_layer_input)
        current_input = self.input_layer.evaluate(current_input)
        for layer in self.hidden_layers:
            every_layer_input.append(current_input)
            current_input = layer.evaluate(current_input)

        every_layer_input.append(np.concatenate([current_input, [1]], axis=0))
        output = self.output_layer.evaluate(current_input, False)
        return (output, every_layer_input)

    def predict(self, x):  # used after the network is tested
        return self.forward(x)[0]

    def error_function(self, x, x_actual):
        return np.linalg.norm(x-x_actual)

    def error_function_gradient(self, x: np.array, x_actual: np.array):
        return 2 * (x - x_actual)

    def get_layer_wages(self, layer):
        return layer.get_all_wages()

    def get_all_wages(self) -> List[List[np.array]]:
        #list of wages of neurons of layers
        wages = []
        wages.append(self.get_layer_wages(self.input_layer))
        for layer in self.hidden_layers:
            wages.append(self.get_layer_wages(layer))
        wages.append(self.get_layer_wages(self.output_layer))
        return wages

    def get_layer_neurons(self, layer):
        return layer.neurons

    def get_all_neurons(self) -> List[List[Neuron]]:

        neurons = []
        neurons.append(self.get_layer_neurons(self.input_layer))
        for layer in self.hidden_layers:
            neurons.append(self.get_layer_neurons(layer))
        neurons.append(self.get_layer_neurons(self.output_layer))
        return neurons

    def backward(self, x, y, predicted, all_inputs_reversed):
        neurons = self.get_all_neurons()
        print(all_inputs_reversed)
        dsums = []  # list of all dsums reversed
        actual = y
        #print(self.error_function(actual, predicted))
        current_dsum = self.error_function_gradient(predicted, actual)
        #print(current_dsum)
        dsums.append([current_dsum])
        previous_layer = self.output_layer
        for indx, layer in enumerate(reversed(self.hidden_layers)):  # propagation from last to first
            current_gradient = layer.propagate_backward_dsum(
                current_dsum, previous_layer
            )
            dsums.append(current_gradient)
            previous_layer = layer
        current_gradient = self.input_layer.propagate_backward_dsum(
            current_dsum, previous_layer
        )
        dsums.append(current_gradient)
        dsums = list(reversed(dsums))
        #all_inputs_reversed = list(reversed(all_inputs_reversed))
        for inx, (layer_neurons, layer_dsum) in enumerate(zip(neurons, dsums)):
            for neuron, dsum in zip(layer_neurons, layer_dsum):
                neuron.wages = neuron.wages - self.LR * dsum * all_inputs_reversed[inx]

    def train(self, x_set: np.array[np.array], y_set, iters):

        for i in range(0, iters):
            print(f":{i}")
            for x, y in zip(x_set, y_set):
                print(f"np.array(x): {np.array([x])}")
                predicted, all_inputs = self.forward(np.array(x))
                self.backward([x], [y], predicted, all_inputs)

###############################################################################
if __name__ == "__main__":
    # TODO: tu prosze podac pierwsze cyfry numerow indeksow
    # JG 324960
    # KK 318380
    p = [3, 7]

    L_BOUND = -5
    U_BOUND = 5

    def q(x: np.array):
        return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

    def q1(x: np.array):
        return np.sign(x)

    def q2(x):
        return 100 * np.sin(x)

    def q3(x):
        return x

    def q4(x):
        return x/x

    def q5(x):
        return np.abs(x)

    x = np.linspace(L_BOUND, U_BOUND, 100)
    converted_x = np.array([np.array([xx]) for xx in x])
    y = q3(x)
    converted_y = np.array([np.array([yy]) for yy in y])

    # np.random.seed(1)

    # currently there is an error with vector input - function with input_dimentionality > 1
    # and with number of layers > 2
    NUMBER_OF_LAYERS = 2
    nn = DlNet(converted_x, converted_y, NUMBER_OF_LAYERS, input_dimentionality=1, HIDDEN_L_SIZE=2)
    nn.train(converted_x, converted_y, 100)

    yh = []  # ToDo tu umiesciÄ wyniki (y) z sieci

    for x_val in x:
        yh.append(nn.predict(x_val)[0])
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-5, 5, -100, 100])
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    print(y)
    print(yh)
    print(nn.output_layer.neurons[0].wages)
    plt.plot(x, y, 'r')
    plt.plot(x, yh, 'b')

    plt.savefig('foo.png')
    plt.show()
