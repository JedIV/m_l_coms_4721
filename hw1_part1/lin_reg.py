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


