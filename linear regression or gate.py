import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

bstack=[]
itered=[]
mstack=[]
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        #print(type(points))
        x = points[i, 0].astype(np.int)
        y = points[i, 1]
        #print(type(x))
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        itered.append(i)
        bstack.append(b)
        mstack.append(m)
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

def run():
    points=np.array([[0,1],[1,0]])
    #points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.5
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 100
    print( "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print( "Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    plt.plot(bstack, itered, label='b', lw=2, marker='o')
    plt.plot(mstack,itered , label='m', lw=2, marker='s')
    plt.xlabel('m   b')
    plt.ylabel('iteration')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


run()