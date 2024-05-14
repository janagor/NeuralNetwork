#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

# f logistyczna jako przyklad sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

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
        wages_scale_factor=None, wage_value: float = None, include_bias=True # currently we use wage_value as determiner
    ):
        # size should not include BIAS - BIAS is automaticaly added
        self.sums = None
        self.size = size
        self.activation_funcion = activation_function
        if not wages_scale_factor:
            self.wages_scale_factor = 1/np.sqrt(self.size)
        self.activation_function = activation_function  # for example sigmoid
        self.initiate_wages(wage_value)  # they include BIAS

    def initiate_wages(self, wage_value=None):
        if wage_value == None:
            self.wages = self.wages_scale_factor * np.random.uniform(
                -self.size, self.size, self.size+1 # there is bias
            )  # last one is bias. At the start it will be equal 1
            self.wages[-1] = 1
        elif wage_value == 0:
            self.wages = wage_value * np.ones(self.size) # no bias

    def evaluate(
        self, input: np.array, activation_function_included=True
    ) -> float:
        result = np.dot(input, self.wages)
        self.sums = result
        if activation_function_included:
            return self.activation_function(result)
        return result


###############################################################################
class Layer:
    def __init__(
        self, input_size, neurons_num,
        activation_function=None, input_wage=None, include_bias=True
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
        cr = []
        for indx, neuron in enumerate(self.neurons):
            sum = np.zeros(len(next_results))
            for result, neuron in zip(next_results, next_layer.neurons):
                sum += result * neuron.wages[indx]
            current_neuron = self.neurons[indx]
            cr.append(sum*d_sigmoid(current_neuron.sums))
        return cr

    def evaluate(
        self, input: np.array, activation_function_included=True
    ) -> np.array:
        temp = list(map(
            lambda neuron: neuron.evaluate(
                input, activation_function_included
            ),
            self.neurons
        ))

        temp = np.array(temp)

        return temp


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
        self.output_layer = Layer(self.HIDDEN_L_SIZE, 1, None, 0, False)

        self.hidden_layers = []
        if number_of_layers > 2:
            for i in range(number_of_layers-2):
                self.hidden_layers.append(Layer(
                    self.HIDDEN_L_SIZE, self.HIDDEN_L_SIZE, activation_function
                ))

        #quality measurement variables
        self.mse = []
        self.mse_final = 0
        self.mae = []
        self.mae_final = []
        self.y_true_arr = []
        self.y_pred_arr = []

    def forward(self, x: np.array) -> Tuple[np.array, List[np.array]]:
        every_layer_input = []
        #new
        current_input = x
        #add
        every_layer_input.append(
            np.concatenate([current_input, np.array([1])])
        )
        #new
        current_input = self.input_layer.evaluate(
            np.concatenate([current_input, [1]], axis=0)
        )
        #add
        every_layer_input.append(np.concatenate([current_input, [1]], axis=0))
        for layer in self.hidden_layers:  # TODO: concatenation for hidden layers
            #new
            current_input = layer.evaluate(current_input)
            #add
            every_layer_input.append(current_input)

        every_layer_input[-1] = every_layer_input[-1][:-1] # usunięcie ostatniego niepotrzebnego elementu
        output = self.output_layer.evaluate(
            current_input,
            False
        )
        # print(output)
        # breakpoint()
        return (output, every_layer_input)

    def predict(self, x):  # used after the network is tested
        return (self.forward(x))[0]  # returns just output

    def error_function(self, x, x_actual):
        return (x-x_actual) ** 2

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

    def backward(self, x, y, predicted, all_inputs):
        neurons = self.get_all_neurons()

        dsums = []  # list of all dsums reversed
        actual = y

        current_dsum = self.error_function_gradient(predicted, actual)

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
        for inx, (layer_neurons, layer_dsum) in enumerate(zip(neurons, dsums)):
            for neuron, dsum in zip(layer_neurons, layer_dsum):
                neuron.wages = neuron.wages - self.LR * dsum * all_inputs[inx]

    def train(self, x_set: np.array[np.array], y_set, iters):
        # prev_average = 100000000
        for i in range(0, iters):
            print(f":{i}")
            # average = 0
            for x, y in zip(x_set, y_set):
                predicted, all_inputs = self.forward(np.array(x))
                self.backward(x, y, predicted, all_inputs)
                # average += self.error_function(y, predicted)
                self.error_save(y, predicted)
            self.quality_measure()
            # average = average/iters
            # if average[0] < prev_average:
            #     print(average[0])
            #     prev_average = average[0]

    def error_save(self, y_true, y_pred):
        self.y_true_arr.append(y_true)
        self.y_pred_arr.append(y_pred)

    def quality_measure(self):
        y_true = self.y_true_arr
        y_pred = self.y_pred_arr
        self.mse.append(mean_squared_error(y_true=y_true, y_pred=y_pred))
        self.mse_final = self.mse[-1]
        self.mae.append(mean_absolute_error(y_true=y_true, y_pred=y_pred))
        self.mae_final = self.mae[-1]

    def get_mse_final(self):  # getter for mean square error calculated in train funciton (returned newest value)
        return self.mse_final
    
    def get_mae_final(self):   # getter for mean absolute error calculated in train funciton (returned newest value)
        return self.mae_final
    
    def get_mse_array(self):  # getter for list of values of mean square error calculated in train funciton (returned newest value)
        return self.mse
    
    def get_mae_array(self):   # getter for list of values of mean absolute error in train funciton (returned newest value)
        return self.mae
        

###############################################################################
if __name__ == "__main__":
    # TODO: tu prosze podac pierwsze cyfry numerow indeksow
    # JG 324960
    # KK 318380
    p = [0, 0]

    L_BOUND = -5
    U_BOUND = 5

    def q(x: np.array):
        return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

    def q1(x: np.array):
        return np.sign(x)

    def q2(x):
        return np.sin(x)

    def q3(x):
        return x

    def q4(x):
        return x/x

    def q5(x):
        return np.abs(x)

    x = np.linspace(L_BOUND, U_BOUND, 100)
    converted_x = np.array([np.array([xx]) for xx in x])
    y = q(x)
    converted_y = np.array([np.array([yy]) for yy in y])

    np.random.seed(1)

    # currently there is an error with vector input - function with input_dimentionality > 1
    # and with number of layers > 2
    NUMBER_OF_LAYERS = 2
    ITERATIONS = 1500
    nn = DlNet(converted_x, converted_y, NUMBER_OF_LAYERS, input_dimentionality=1, HIDDEN_L_SIZE=5)
    nn.train(converted_x, converted_y, ITERATIONS)
    print(nn.get_all_wages())
    # print(nn.get_layer_wages(nn.output_layer))
    # breakpoint()
    yh = []  # ToDo tu umiesciÄ wyniki (y) z sieci

    for x_val in x:
        # print(type(x_val))
        # breakpoint()
        yh.append(nn.predict(np.array([x_val]))[0])
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.axis([-5, 5, -100, 100])
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # print(y)
    # print(yh)
    # print(nn.output_layer.neurons[0].wages)
    plt.plot(x, y, 'r')
    plt.plot(x, yh, 'b')

    plt.savefig('foo.png')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    # ax.axis([-5, 5, -100, 100])
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')

    # qulaity measurement values
    iters = np.arange(1, len(nn.get_mse_array())+1, 1)
    plt.plot(iters, nn.get_mse_array(), 'r')
    
    ax = fig.add_subplot(1, 2, 2)
    # ax.axis([-5, 5, -100, 100])
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    plt.plot(iters, nn.get_mae_array(), 'b')
    # plt.savefig('quality.png')
    plt.show()
