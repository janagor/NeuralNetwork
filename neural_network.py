#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: RafaĹ Biedrzycki
Kodu tego mogÄ uĹźywaÄ moi studenci na Äwiczeniach z przedmiotu WstÄp do Sztucznej Inteligencji.
Kod ten powstaĹ aby przyspieszyÄ i uĹatwiÄ pracÄ studentĂłw, aby mogli skupiÄ siÄ na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakoĹci programowania w Pythonie, nie jest rĂłwnieĹź wzorem programowania obiektowego, moĹźe zawieraÄ bĹÄdy.

Nie ma obowiÄzku uĹźywania tego kodu.
"""
from __future__ import annotations
import numpy as np

#ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [3,7]

L_BOUND = -5
U_BOUND = 5

def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


# f logistyczna jako przykĹad sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)
     
#f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

#pochodna f. straty
def d_nloss(y_out, y):
    return 2*( y_out - y )


###############################################################################
# I want to create a network for input data dimention equal 2 and number of inner layers greater than two (it will allow me to go into grater sizes later)
# assumptions: all inner layers have the same number of neurons
class Neuron:
    def __init__(self, x:np.array, wages:np.array=None, activation_function = sigmoid, wages_scale_factor=None):
        self.BIAS = 1
        if not wages_scale_factor:
            self.wages_scale_factor = 1/np.sqrt(self.size)
        self.x = x
        self.y = None # skalar output
        self.size = x.size
        if not wages_scale_factor:
            self.wages_scale_factor = 1/np.sqrt(self.size)
        self.wages = wages # niech len(wages) = len(x) + 1 # because there is a bias at start for activation function
        self.activation_function = activation_function # for example sigmoid
        if not wages:
            self.initiate_wages()

    def initiate_wages(self):
        self.wages = self.wages_scale_factor * np.random.uniform(-self.size, self.size)

    def calculate_output() -> float:
        activate_value = (self.x * self.wages).sum() + self.BIAS
        return self.activation_function(activate_value)


class Layer:
    def __init__(self, size, else):
        self.size = size

###############################################################################
class DlNet:
    def __init__(self, x, y, number_of_layers, input_dimentionality=2, HIDDEN_L_SIZE = 9):
        self.x = x
        self.y = y
        self.y_out = 0
        self.input_dimentionality = input_dimentionality
        self.HIDDEN_L_SIZE = 9 # number of neurons in a single layer
        self.LR = 0.003 # learning rate
        self.input_layer = Layer(self.HIDDEN_L_SIZE)
        self.output_layer = Layer(self.HIDDEN_L_SIZE)
        self.hidden_layers = []
        if number_of_layers> 2:
            for i in range(number_of_layers-2):
                self.hidden_layers.append(Layer(self.HIDDEN_L_SIZE))


    def forward(self, x): # used by train when we want to achieve result value of the aproctimator and also used by predict simply to get results
        
#ToDo

    def predict(self, x): # used after the network is tested
        #ToDo
        return RESULT_TODO

    def backward(self, x, y): # I suppose it is used when training to update and include impact of a given x on network hiperparams (thus also results)
#ToDo
        pass

    def train(self, x_set, y_set, iters):
        for i in range(0, iters):
            pass
        pass
#ToDo

###############################################################################
if __name__="__main__":
    nn = DlNet(x,y)
    nn.train(x, y, 15000)

    yh = [] #ToDo tu umiesciÄ wyniki (y) z sieci

    import matplotlib.pyplot as plt


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x,y, 'r')
    plt.plot(x,yh, 'b')

    plt.show()
