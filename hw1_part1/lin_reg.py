####################
#lin_reg.py
#Homework 1
#Jed Dougherty
#File to read in csv
#and create basic
#plots
####################

import pandas as p
from matplotlib import pyplot as plt


#reads in file and names extra columns
girls = p.read_csv('girls_train.csv',
          names = ['age', 'height','x','y'])

#sets up plotting canvas
fig = plt.figure()

#creates plot of age v height
girls.plot('age','height',
    marker='o', color='r', ls='')

#title and axis options
fig.suptitle('CDC Data of Age V Height', fontsize=20)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Height', fontsize=16)
plt.xlim([1, 9])

#outputs to png file
fig.savefig('age_v_height.png')


# in matrix form this is as follows:
# def gradientDescent(alpha = .05, x, y, eps = .001, iterations = 1500) {
    # m = 0
    # n = x.shape[0]



# grad <- function(x, y, theta) {
  # gradient <- (1/m)* (t(x) %*% ((x %*% t(theta)) - y))
  # return(t(gradient))
# }

# define gradient descent update algorithm
# grad.descent <- function(x, maxit){
    # theta <- matrix(c(0, 0), nrow=1) # Initialize the parameters
 
    # alpha = .05 # set learning rate
    # for (i in 1:maxit) {
      # theta <- theta - alpha  * grad(x, y, theta)   
    # }
 # return(theta)
# }
