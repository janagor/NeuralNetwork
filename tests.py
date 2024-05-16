#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Union, List
from neural_network import Neuron, Layer, DlNet

# TODO: tu prosze podac pierwsze cyfry numerow indeksow
# JG 324960
# KK 318380
p = [0, 0]

def q(x: np.array):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

def test_with_all_params(iterations: int, hidden_size: int): # currently it only involves NUMBER_OF_LAYERS = 2

    L_BOUND = -5
    U_BOUND = 5

    mse_sum = 0
    mae_sum = 0
    r2_sum = 0

    x = np.linspace(L_BOUND, U_BOUND, 100)
    converted_x = np.array([np.array([xx]) for xx in x])
    y = q(x)
    converted_y = np.array([np.array([yy]) for yy in y])
    NUMBER_OF_LAYERS = 2 # only 2 is available
    nn = DlNet(converted_x, converted_y, NUMBER_OF_LAYERS, input_dimentionality=1, HIDDEN_L_SIZE=hidden_size)
    nn.train(converted_x, converted_y, iterations)

    yh = [] 

    for x_val in x:
        yh.append(nn.predict(np.array([x_val]))[0])

    for y_true, y_pred in zip(y, yh):
        nn.values_save(y_true=y_true, y_pred=y_pred)
    nn.quality_measure()
    # print(f"Iters: {iterations}, Hidden layers: {NUMBER_OF_LAYERS-1}, Hl size: {hidden_size}")
    # print(f"MSE: {nn.mse}")
    # print(f"MAE: {nn.mae}")
    # print(f"R2: {nn.r2}") 
    mse_sum += nn.mse
    mae_sum += nn.mae
    r2_sum += nn.r2
    return (mse_sum, mae_sum, r2_sum)


def tests_multiple_instances(iterations: int=1000, hidden_size: int=5, tests_number: int=1, seed: int=1) -> Tuple[float, float, float]:
    print(iterations)
    np.random.seed(seed)
    mse_all = np.zeros(tests_number)
    mae_all = np.zeros(tests_number)
    r2_all = np.zeros(tests_number)
    for test in range(0, tests_number):
        mse, mae, r2 = test_with_all_params(iterations, hidden_size)
        mse_all[test] = mse
        mae_all[test] = mae
        r2_all[test] = r2

    print("FINAL RESULTS:")
    print("----------------------------")
    print(f"Iters: {iterations}, Hidden layers: 2, Hl size: {hidden_size}") # currently only 2 hidden layers
    print("----------------------------")
    print(f"MSE AVG: {np.average(mse_all)}")
    print(f"MSE MAX: {np.max(mse_all)}")
    print(f"MSE MIN: {np.min(mse_all)}")
    print(f"MSE STD: {np.std(mse_all)}")
    print("----------------------------")
    print(f"MAE AVG: {np.average(mae_all)}")
    print(f"MAE MAX: {np.max(mae_all)}")
    print(f"MAE MIN: {np.min(mae_all)}")
    print(f"MAE STD: {np.std(mae_all)}")
    print("----------------------------")
    print(f"R2 AVG: {np.average(r2_all)}")
    print(f"R2 MAX: {np.max(r2_all)}")
    print(f"R2 MIN: {np.min(r2_all)}")
    print(f"R2 STD: {np.std(r2_all)}")
    print("----------------------------")

    # for latex copy/paste only
    # print(str(hidden_size) + " & \\num{" + str(("%.3f" % (np.average(mse_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.max(mse_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.min(mse_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.std(mse_all)))) + "}" + "\\\\ \\hline")
    # print(str(hidden_size) + " & \\num{" + str(("%.3f" % (np.average(mae_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.max(mae_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.min(mae_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.std(mae_all)))) + "}" + "\\\\ \\hline")
    # print(str(hidden_size) + " & \\num{" + str(("%.3f" % (np.average(r2_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.max(r2_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.min(r2_all)))) + "}" + " & \\num{" + str(("%.3f" % (np.std(r2_all)))) + "}" + "\\\\ \\hline")
    
    return np.average(mse_all), np.average(mae_all), np.average(r2_all)


def test_iterations(iterations_instances: np.array, hidden_size: int=5, tests_number: int=1, seed: int=1):
    results = []

    for iteration in iterations_instances:
        result = tests_multiple_instances(iteration, hidden_size, tests_number, seed)
        results.append(result)
    x = iterations_instances
    y_mse, y_mae, y_r2 = zip(*results)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    plt.plot(x, y_mse)
    plt.title("Mean Square Error (MSE) value\ndepending on the number of iterations.")
    plt.xlabel("number of iterations")
    plt.ylabel("mse")
    plt.savefig('mse_iterations.png')
    plt.show()

    plt.plot(x, y_mae)
    plt.xlabel("number of iterations")
    plt.ylabel("mae")
    plt.title("Mean Absolute Error (MAE) value\ndepending on the number of iterations.")
    plt.savefig('mae_iterations.png')
    plt.show()

    plt.plot(x, y_r2)
    plt.xlabel("number of iterations")
    plt.ylabel("mse")
    plt.title("R-squared (R2) value\ndepending on the number of iterations.")
    plt.savefig('r2_iterations.png')
    plt.show()

def test_hidden_layer_size(hidden_layer_size_instances: np.array, iterations: int = 400, tests_number: int=1, seed: int=1):
    results = []

    for hl_size in hidden_layer_size_instances:
        result = tests_multiple_instances(iterations, hl_size, tests_number, seed)
        results.append(result)
    x = hidden_layer_size_instances
    y_mse, y_mae, y_r2 = zip(*results)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    plt.xticks(ticks=hidden_layer_size_instances)
    plt.plot(x, y_mse)
    plt.title("Mean Square Error (MSE) value\ndepending on the hidden layer size.")
    plt.xlabel("hidden layer sizes")
    plt.ylabel("mse")
    plt.savefig('mse_hl_size.png')
    plt.show()


    plt.xticks(ticks=hidden_layer_size_instances)
    plt.plot(x, y_mae)
    plt.xlabel("hidden layer sizes")
    plt.ylabel("mae")
    plt.title("Mean Absolute Error (MAE) value\ndepending on the hidden layer size.")
    plt.savefig('mae_hl_size.png')
    plt.show()


    plt.xticks(ticks=hidden_layer_size_instances)
    plt.plot(x, y_r2)
    plt.xlabel("hidden layer sizes")
    plt.ylabel("mse")
    plt.title("R-squared (R2) value\ndepending on the hidden layer size.")
    plt.savefig('r2_hl_size.png')
    plt.show()

if __name__ == "__main__":
    iterations = 400
    # iterations = np.array([100, 200])
    hidden_size = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 20])
    tests_number = 5
    seed = np.random.randint(0, 10000)
    test_hidden_layer_size(hidden_size, iterations, tests_number, seed=seed)
