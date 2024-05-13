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

import numpy as np

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

class Neuron:
	def __init__(self, wage = None, size = int, activation_function=sigmoid):
		self.inputs_number = size	# casue of bias
		self.wages_scale_factor = 1/np.sqrt(self.inputs_number)
		self.wages = []
		self.activation_function = activation_function  # for example sigmoid
		self.initiate_wages(wage)  # they include BIAS
		self.output = 0

	def initiate_wages(self, wage):
		if wage == None:
			self.wages = self.wages_scale_factor * np.random.uniform(
				-self.inputs_number, self.inputs_number, self.inputs_number+1
			)  # last one is bias
			self.wages[-1] = 1
		else:
			self.wages = wage * np.ones(self.inputs_number+1)

	def evaluate(self, input: np.array) -> float:
		result = (input * self.wages[:-1]).sum() + self.wages[-1]
		self.output = self.activation_function(result)
		return self.output

class DlNet:
	def __init__(self, x, y, HIDDEN_L_SIZE = 9, LEARNING_RATE = 0.003):
		self.x = x
		self.y = y
		self.y_out = 0
		self.bias = 1

		self.hidden_l_size = HIDDEN_L_SIZE
		self.lr = LEARNING_RATE

		self.input_neuron = Neuron(size = 1, wage = 1)
		self.hidden_neurons = []
		for _ in range(0, self.hidden_l_size):
			self.hidden_neurons.append(Neuron(size = 1))
		self.output_neuron = Neuron(size = self.hidden_l_size)

		#ToDo        


	def forward(self, x):  
		#self.inputs = ....
		self.input_neuron.evaluate(np.array(x))
		hn_results = np.array(float, ndmin=1)
		hn_results = []
		for hn in self.hidden_neurons:
			hn_results.append((hn.evaluate(np.array(self.input_neuron.output))))
		self.output_neuron.evaluate(np.array(hn_results))
		#ToDo        

	def predict(self, x):    
		self.forward(x)  
		return self.output_neuron.output

	def backward(self, x, y):
		output_error = d_nloss(self.output_neuron.output, y)
		output_delta = output_error * d_sigmoid(self.output_neuron.output)

		hidden_errors = np.dot(output_delta, np.array(self.output_neuron.wages).T)
		hidden_outputs = []
		for hn in self.hidden_neurons:
			hidden_outputs.append(hn.output)
		hidden_outputs.append(self.output_neuron.wages[-1] * x)

		hiddes_tmp = []
		for hr in hidden_outputs:
			hiddes_tmp.append(d_sigmoid(hr))
		hidden_delta = hidden_errors * hiddes_tmp
		
		self.output_neuron.wages -= self.lr * np.dot(hidden_outputs, output_delta)
	
		new_hidden_weights_factors = self.lr * np.dot(self.input_neuron.output, hidden_delta)
		i = 0
		for hn in self.hidden_neurons:
			hn.wages -= new_hidden_weights_factors[i]
			i += 1 

	def train(self,x_set, y_set, iters):    
		for _ in range(0, iters):
			# print(f"Iter: {_} / {iters}")
			for x, y in zip(x_set, y_set):
				self.forward(x)
				self.backward(x, y)
			a = self.output_neuron.output - y
			print(f"Err: {a} / {iters}")
		#ToDO

def main():
	#ToDo tu prosze podac pierwsze cyfry numerow indeksow
	# JG 324960
	# KK 318380
	p = [0, 0]

	L_BOUND = -5
	U_BOUND = 5

	def q(x):
		return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))
	
	def q1(x: np.array):
		return np.sign(x)
	
	def q2(x):
		return np.sin(0.5*x)

	def q3(x):
		return 4*x

	x = np.linspace(L_BOUND, U_BOUND, 100)
	y = q1(x)

	nn = DlNet(x,y, HIDDEN_L_SIZE=5, LEARNING_RATE=0.001)
	nn.train(x, y, 1000)

	yh = [] #ToDo tu umiesciÄ wyniki (y) z sieci

	for x_val in x:
		yh.append(nn.predict(x_val))

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

if __name__ == "__main__":
	main()

