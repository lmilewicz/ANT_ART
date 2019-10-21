# -*- coding: utf-8 -*-
import numpy as np
import random

######  Step 0  ######

# Maximum number of ants:
ant_number = 5

# Maximum number of iterations:
Mn = 100

# Side length of local region:
s = 5

# Maximum speed of ants moving
v_max = 10

# Alpha & betha parameters
alpha = 0.8 
betha = 0.9

# Quantity of objects
N = 100

# x dimension
X = 100

# y dimension
Y = 100

# create a space
space = np.zeros((X, Y))

######  Step 1  ######
#Project the data onthe plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random
for object_index in range(N):
    x = random.randint(0,X-1)
    y = random.randint(0,Y-1)
    space[x,y]=1


######  Step 2  ######

#for i in range(1, Mn):
#    for j in range(1, ant_number):
        #2.1
        #2.2 
        #2.3
        

######  Step 3  ######

