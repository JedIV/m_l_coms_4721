####################
#lin_reg.py
#Homework 1
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
girls = np.genfromtxt('girls_train.csv', delimiter=',',usecols = (0,1))
girls_test = np.genfromtxt('girls_test.csv', delimiter=',',usecols = (0,1))


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


x = np.column_stack((np.ones(len(girls)),girls[:,0]))
y = girls[:,1]
m, n = np.shape(x)
numIterations= 1500
alpha = 0.05
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

def approx_outputs(inputs, slope, intercept):
    output = np.dot(slope,inputs) + intercept
    return output

approx_height = approx_outputs(girls[:,0],theta[1],theta[0])

def mse(inputs, outputs, slope, intercept):
    exp_output = approx_outputs(inputs,slope,intercept)
    sqError = (outputs - exp_output)**2
    return np.mean(sqError)

calc = mse(girls[:,0],girls[:,1],theta[1],theta[0])

##Preps data for 3d plotting
xi = np.linspace(-1,1,100)
yi = np.linspace(-3,3,100)
X, Y = np.meshgrid(xi, yi)
Z = np.zeros((len(xi),len(yi)))
for i in range(0,len(xi)):
    for j in range(0,len(yi)):
        inner = X[i,j]
        after = Y[i,j]
        Z[i][j] = mse(girls[:,0], girls[:,1], inner, after)

#CONTOUR PLOT
fig_cont = plt.figure()

plt.contour(Y, X, np.log(Z),cmap = 'copper')
plt.plot(theta[0],theta[1], marker='o', color='r', ls='')

fig_cont.suptitle('CDC Data of Age V Height', fontsize=20)
plt.xlabel('Beta 1', fontsize=16)
plt.ylabel('Beta 0', fontsize=16)
plt.show(fig_cont)

# sets up plotting canvas
fig_lines = plt.figure()

# DOT PLOT WITH FITTED LINE
# creates plot of age v height
plt.plot(girls[:,0],girls[:,1], marker='o', color='r', ls='')
plt.plot(girls[:,0], approx_height, 'b')

# title and axis options
fig_lines.suptitle('CDC Data of Age V Height', fontsize=20)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Height', fontsize=16)
plt.xlim([1, 9])

plt.show(fig_lines)
# outputs to png file

# fig.savefig('age_v_height.png')



#3D Plot
fig_3d = plt.figure(figsize=(9,7))

ax = fig_3d.add_subplot(1,1,1, projection='3d')

ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir='z', offset=-3*np.pi, cmap='coolwarm')
cset = ax.contour(X, Y, Z, zdir='x', offset=-1.5, cmap='coolwarm')
cset = ax.contour(X, Y, Z, zdir='y', offset=3.5, cmap='coolwarm')

ax.set_xlim3d(-1.5, 1);
ax.set_ylim3d(-3, 3.5);
ax.set_zlim3d(-3*np.pi, 90);

fig_3d.suptitle('CDC Data of Age V Height', fontsize=20)
ax.set_xlabel('Beta 1 ', fontsize=16)
ax.set_ylabel('Beta 0', fontsize=16)
ax.set_zlabel('Error', fontsize=16)

plt.show(fig_3d)

#############Prediction for a 4.5 year old girl
checked_height = approx_outputs(4.5,theta[1],theta[0])

#############Prediction of test data
test_mse = mse(girls_test[:,0],girls_test[:,1],theta[1],theta[0])
