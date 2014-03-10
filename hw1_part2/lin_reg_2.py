####################
#lin_reg_2.py
#Homework 1 part 2
#Jed Dougherty
#File to read in csv
#and create basic
#plots
####################

import numpy as np
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
import random
from numpy.random import uniform
from mpl_toolkits.mplot3d.axes3d import Axes3D

#reads in file and names extra columns
girls_a_w_h = np.genfromtxt('hw1_all/girls_age_weight_height_2_8.csv', delimiter=',')

#creates a normalized version of the data
girls_norm = np.copy(girls_a_w_h)

#normalizer function
def normalizer(x = 0):
    mean = np.mean(girls_a_w_h[:,x])
    sd = np.std(girls_a_w_h[:,x])
    norm = (girls_a_w_h[:,x] - mean)/sd
    return norm


#run on each column
girls_norm[:,0] = normalizer(0)
girls_norm[:,1] = normalizer(1)
girls_norm[:,2] = normalizer(2)


# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def normalEquation(X,y):
    theta = y * X.T * np.linalg.inv(X * X.T)
    return theta


#setting up data to be passed through
x = np.column_stack((np.ones(len(girls_norm)),girls_norm[:,range(0,2)]))
y = girls_norm[:,2]
m, n = np.shape(x)
numIterations= 1500
alpha = 0.05
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

denorm_x = np.column_stack((np.ones(len(girls_a_w_h)),girls_a_w_h[:,range(0,2)]))
denorm_y = girls_a_w_h[:,2]

#Calculating the normals
exact_theta = np.linalg.inv((denorm_x.T).dot(denorm_x)).dot(denorm_x.T).dot(y)
print(exact_theta)


