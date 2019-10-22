# -*- coding: utf-8 -*-
import numpy as np
import random
import math

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

######  Step 0  ######

# Maximum number of ants:
ant_number = 5

# Maximum number of iterations:
Mn = 2

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

# side of local regions
s = 10

# declare v
v = 34
vmax = 34

# function
def sigmoid_function(x,betha):
  return 1 / (1+ math.exp(- x * betha))


######  Step 1  ######
#Project the data onthe plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random
for object_index in range(N):
    x = random.randint(0,X-1)
    y = random.randint(0,Y-1)
    space[x,y]=1


[nonzero_x,nonzero_y] = np.nonzero(space)
print(nonzero_x)
print(nonzero_y)

######  Step 2  ######

for i in range(1, Mn):
    for j in range(1, ant_number):
        for n in range(N-2):
            print(i,j,n)
            oix = nonzero_x[n]
            oiy = nonzero_y[n]
            mask_min_x=max(0,oix-s)
            mask_min_y=max(0,oiy-s)
            mask_max_x=min(oix+s,X)
            mask_max_y=min(oiy+s,Y)
##            print(oix,oiy,'sd')
##            print(mask_min_x)
##            print(mask_min_y)
##            print(mask_max_x)
##            print(mask_max_y)
            neigh_sum = 0
            for n2 in range(N-2):
                ojx = nonzero_x[n2]
                ojy = nonzero_y[n2]                  
                if ((ojx != oix) or (oiy != ojy)) and (ojx > mask_min_x) and (ojx < mask_max_x) and (ojy > mask_min_y) and (ojy < mask_max_y) :
                    object=[oix,oiy]
                    neighbour=[ojx,ojy]
##                    print(neighbour)
                    doioj = dot(object,neighbour)/norm(object)/norm(neighbour)
##                    print(foi)
                    neigh_sum = neigh_sum + (1-(doioj/(alpha*(1+((v-1)/vmax)))))
            foi = max(0,neigh_sum/(s*s))
            sigm_fun = sigmoid_function(foi,betha)
            rand=random.random()
            picking_prob = 1 - sigmoid_function(foi,betha)
            if (picking_prob > rand) and space[oix,oiy] != 2 :
                space[oix,oiy] = 2

            droping_prob = sigmoid_function(foi,betha)
            print(sigm_fun)
        

######  Step 3  ######

