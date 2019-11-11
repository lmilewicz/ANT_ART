# -*- coding: utf-8 -*-
import numpy as np
import random

from CDbw import CDbw
from Ant import (Ant, dataObject)

from otherFunctions import (sigmoid, sim)


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


######  Step 1  ######
#Project the data on the plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random


objects = []

for i in range(N):
    x = random.randint(0,X-1)
    y = random.randint(0,Y-1)
    objects.append(dataObject([x, y]))

U = []
U.append(objects[0:5])
U.append(objects[5:10])

print(CDbw(U))

AntColony = []
for i in range(ant_number):
    AntColony.append(Ant(objects[i]))


######  Step 2  ######

for i in range(Mn-1): # for i = 1, 2 ..., Mn
    for j in range(ant_number-1): # for j = 1, 2 ..., ant_number
        ant = AntColony[j]
        
        oi = ant.dataObject
        oixy = oi.coord
        oix = oixy[0]
        oiy = oixy[1]
        
        mask_min_x=max(0,oix-s)
        mask_min_y=max(0,oiy-s)
        mask_max_x=min(oix+s,X)
        mask_max_y=min(oiy+s,Y)

        neigh_sum = 0
        for oj in objects:
            ojxy = oj.coord
            ojx = ojxy[0]
            ojy = ojxy[1]                  
            if oi != oj and ojx > mask_min_x and ojx < mask_max_x and ojy > mask_min_y and ojy < mask_max_y :
                neigh_sum = neigh_sum + (1-(1-sim(oixy,ojxy)/(alpha*(1+((v-1)/vmax)))))

        foi = max(0, neigh_sum/(s*s))
        rand = random.random()
        
        if ant.label == 'Unloaded':
            picking_prob = 1 - sigmoid(foi, beta)
            if picking_prob > rand and ant.dataObject.label == 'Unloaded':
                ant.loadObject()
                ant.moveAnt()
            else:
                ant.setObject(objects[random.randint(0,len(objects))])
        else:
            droping_prob = sigmoid(foi, beta)
            if droping_prob > rand:
                ant.dropObject()
                ant.setObject(objects[random.randint(0,len(objects))])
            else:
                ant.moveAnt()
        

######  Step 3  ######

neigh_num = 0
neigh_min = 2

for oi in objects:
    for oj in objects:
        if oi != oj:
            if sim(oi.coord, oj.coord) > 0.1: # check neighbours
                neigh_num = neigh_num + 1
    if neigh_num < neigh_min:
        oi.label = 'Outlier'
    else:
         # give this object a cluster sequance number and recursively label 
        #the same sequence numer to those objects who is the neighbors of this 
        #object within local region, then obtain the number of clusters c
        oi.cluster = 1
        c = 1
            

######  Step 4  ######

for i in range(c):
    #4.1 Compute the mean of the cluster and find four representative points
    #by scanning the cluster in the plane from different direcion of x-axis and y-axis
    print()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
