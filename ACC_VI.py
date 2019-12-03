# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import sys
import matplotlib.pyplot as plt

from CDbw import CDbw
from Ant import (Ant, cluster)

from otherFunctions import (sigmoid, sim, printAntColony, printObjects, getMove, returnObjects, convertToArray)


#%%     ######  Step 0  ######

# Maximum number of ants:
ant_number = 50

# Maximum number of iterations:
Mn = 200

# Side length of local region:
s = 10

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


# declare v
v = 10


#%%
#Project the data on the plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random

print('######  Step 1  ######')

start = time.time()

objects = returnObjects(X, Y, N, 'iris')

AntColony = []
for i in range(ant_number):
    AntColony.append(Ant(objects[random.randint(0,len(objects)-1)]))

#printAntColony(AntColony)
printObjects(objects)

end = time.time()
print('Execution time: %0.2f' % (end - start))

#%%
print('\n######  Step 2  ######')

start = startX = time.time()

for i in range(Mn): # for i = 1, 2, ..., Mn
    for j in range(ant_number): # for j = 1, 2, ..., ant_number
        ant = AntColony[j]

        oi = ant.dataObject
        mask_min = np.maximum(0, oi.coord - s)
        mask_max = np.minimum(oi.coord + s, [X, Y])

        sim_sum = 0
        for oj in objects:
            if oi != oj and np.all(oj.coord > mask_min) and np.all(oj.coord < mask_max):
                sim_sum = sim_sum + (1-(1-sim(oi.coord, oj.coord)/(alpha*(1+((v-1)/v_max)))))

        foi = max(0, sim_sum/(s*s))
        rand = random.random()

        if ant.label == 'Unloaded':
            picking_prob = 1 - sigmoid(foi, beta)
            if picking_prob > rand and ant.dataObject.antLabel == 'Unloaded':
                ant.loadObject()
                ant.move(getMove(v))
            else:
                ant.setObject(objects[random.randint(0,len(objects)-1)])
        else:
            droping_prob = sigmoid(foi, beta)
            if droping_prob > rand:
                ant.dropObject()
                ant.setObject(objects[random.randint(0,len(objects)-1)])
            else:
                ant.move(getMove(v))
    
    if i%(int(Mn/100)) == 0: 
        endX = time.time()
        sys.stdout.write('\rProgress: %0.2f percent. Execution time: %0.2f' % (100*(i+1)/Mn, endX - startX))
        sys.stdout.flush()
        startX = time.time()

#printAntColony(AntColony)
printObjects(objects)

end = time.time()
print('\n\nExecution time: %0.2f' % (end - start))

#%%
print('\n######  Step 3  ######')

start = time.time()

def checkClustersIfEqual(cluster1, cluster2):
    if len(cluster1.objectsList) != len(cluster2.objectsList):
        return False
    else:
        i = 0
        for oi in cluster1.objectsList:
            i_prev = i
            for oj in cluster2.objectsList:
                if oi == oj:
                    i = i + 1
            if i_prev == i:
                return False
        return True


neigh_min = 3
c = 0
clusters = []
outliersList = []

for oi in objects:
    neigh_num = 0
    mask_min = np.maximum(0, oi.coord - s)
    mask_max = np.minimum(oi.coord + s, [X, Y])
    for oj in objects:
        if oi != oj and np.all(oj.coord > mask_min) and np.all(oj.coord < mask_max):
            neigh_num = neigh_num + 1
    if neigh_num < neigh_min:
        oi.setOutlier()
        outliersList.append(oi)
    else:
        clusters.append(cluster(name = c))
        clusters[c].addObject(oi)
        for oj in objects:
            if oj.label != 'Outlier' and oi != oj and np.all(oj.coord > mask_min) and np.all(oj.coord < mask_max):
                clusters[c].addObject(oj)
        c = c + 1

        for i, cl in enumerate(clusters):
            if i < c - 1 and checkClustersIfEqual(cl, clusters[c-1]):
                for oi in clusters[c-1].objectsList:
                    oi.unsetCluster()
                clusters.pop()
                c = c - 1
                break

end = time.time()
print('Execution time: %0.2f' % (end - start))

#%%
print('\n######  Step 4  ######')

start = endX = time.time()

if c > 1:
    U = convertToArray(clusters)
    endX = time.time()
    
    CDbwValue = CDbw(U)

    '''for cl in clusters:
    #4.1 Compute the mean of the cluster and find four representative points
    #by scanning the cluster in the plane from different direcion of x-axis and y-axis
    print(CDbw(cl))
    '''
end = time.time()
print('Execution time: %0.2f. ConvertToArray time: %0.4f' % (end - start, endX - startX))

#%%
print('\n######  Step 5  ######')

start = time.time()

if c > 1 and len(outliersList) > 0:
    CDbwVector = np.zeros(c)
    for i, cl in enumerate(clusters):
        startX = time.time()
        cl.objectsList.append(outliersList[0])

        U = convertToArray(clusters)
        CDbwVector[i] = CDbw(U) - CDbwValue        
        cl.objectsList.remove(outliersList[0])
        endX = time.time()
        sys.stdout.write('\rProgress: %0.2f percent. Execution time: %0.2f' % (100*(i+1)/c, endX - startX))
        sys.stdout.flush()
        
    
    clusters[np.argmax(np.absolute(CDbwVector))].objectsList.append(outliersList[0])
    outliersList.pop(0)
    #print(CDbwVector)

end = time.time()

print('\n\nExecution time: %0.2f' % (end - start))

#clusters[np.maxposition(CDbwVector)]
