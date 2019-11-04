# -*- coding: utf-8 -*-
import numpy as np
import random
import math

from CDbw import CDbw

######  Step 0  ######

# Maximum number of ants:
ant_number = 5

# Maximum number of iterations:
Mn = 2

# Side length of local region:
s = 5

# Maximum speed of ants moving
v_max = 10

# Alpha & beta parameters
alpha = 0.8 
beta = 0.9

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

# functions
def sigmoid_function(x, beta):
    return 1 / (1+ math.exp(-x * beta))

def sim(o_i, o_j):
    return np.sum(o_i*o_j)/np.sqrt(np.sum(o_i*o_i)*np.sum(o_j*o_j))

######  Step 1  ######
#Project the data onthe plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random
for i in range(N):
    x = random.randint(0,X-1)
    y = random.randint(0,Y-1)
    space[x,y]=1

objects = np.transpose(np.nonzero(space))

U = []
U.append(objects[0:5])
U.append(objects[5:10])

print(CDbw(U))



######  Step 2  ######

for i in range(1, Mn):
    for j in range(1, ant_number):
        for oixy in objects:
            #print(i,j,n)
            oix = oixy[0]
            oiy = oixy[1]
            mask_min_x=max(0,oix-s)
            mask_min_y=max(0,oiy-s)
            mask_max_x=min(oix+s,X)
            mask_max_y=min(oiy+s,Y)

            neigh_sum = 0
            for ojxy in objects:
                ojx = ojxy[0]
                ojy = ojxy[1]                  
                if (oix != ojx or oiy != ojy) and ojx > mask_min_x and ojx < mask_max_x and ojy > mask_min_y and ojy < mask_max_y :
                    neigh_sum = neigh_sum + (1-(1-sim(oixy,ojxy)/(alpha*(1+((v-1)/vmax)))))

            foi = max(0, neigh_sum/(s*s))
            sigm_fun = sigmoid_function(foi,beta)
            rand=random.random()
            picking_prob = 1 - sigmoid_function(foi,beta)
            if (picking_prob > rand) and space[oix,oiy] != 2 :
                space[oix,oiy] = 2

            droping_prob = sigmoid_function(foi,beta)
            #print(sigm_fun)
        

######  Step 3  ######

