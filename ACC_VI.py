# -*- coding: utf-8 -*-
import numpy as np
import random

from sklearn import datasets

from CDbw import CDbw
from Ant import (Ant, dataObject, cluster)

from otherFunctions import (sigmoid, sim, printAntColony, printObjects, getMove)


#%%     ######  Step 0  ######

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
s = 5

# declare v
v = 34
vmax = 34


#%%     ######  Step 1  ######
#Project the data on the plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random

print('######  Step 1  ######')

def returnObjects(X, Y, N, dataType):
    objects = []

    if dataType == 'random':
        for i in range(N):
            x = random.randint(0,X-1)
            y = random.randint(0,Y-1)
            objects.append(dataObject([x, y]))
    elif dataType == 'iris':
        iris = datasets.load_iris()
        if N > len(iris.data):
            N = len(iris.data)
        db_iris = iris.data[:N, :2]
        for i in range(N):
            x = (db_iris[i, 0]/max(db_iris[:,0])) * X
            y = (db_iris[i, 1]/max(db_iris[:,0])) * Y
            objects.append(dataObject([x, y]))
    return objects

objects = returnObjects(X, Y, N, dataType='iris')

AntColony = []
for i in range(ant_number):
    AntColony.append(Ant(objects[random.randint(0,len(objects)-1)]))

printAntColony(AntColony)
printObjects(objects)

#%%     ######  Step 2  ######

print('\n######  Step 2  ######')

for i in range(Mn-1): # for i = 1, 2, ..., Mn
    for j in range(ant_number-1): # for j = 1, 2, ..., ant_number
        ant = AntColony[j]

        oi = ant.dataObject
        mask_min = np.maximum(0, oi.coord - s)
        mask_max = np.minimum(oi.coord + s, [X, Y])

        sim_sum = 0
        for oj in objects:
            if oi != oj and np.all(oj.coord > mask_min) and np.all(oj.coord < mask_max):
                sim_sum = sim_sum + (1-(1-sim(oi.coord,oj.coord)/(alpha*(1+((v-1)/vmax)))))

        foi = max(0, sim_sum/(s*s))
        rand = random.random()

        if ant.label == 'Unloaded':
            picking_prob = 1 - sigmoid(foi, beta)
            if picking_prob > rand and ant.dataObject.antLabel == 'Unloaded':
                ant.loadObject()
                ant.move(getMove())
            else:
                ant.setObject(objects[random.randint(0,len(objects)-1)])
        else:
            droping_prob = sigmoid(foi, beta)
            if droping_prob > rand:
                ant.dropObject()
                ant.setObject(objects[random.randint(0,len(objects)-1)])
            else:
                ant.move(getMove())


printAntColony(AntColony)
printObjects(objects)

#%%     ######  Step 3  ######

print('\n######  Step 3  ######')

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

#%%     ######  Step 4  ######

print('\n######  Step 4  ######')

if c > 1:
    CDbwValue = CDbw(clusters)

    '''for cl in clusters:
    #4.1 Compute the mean of the cluster and find four representative points
    #by scanning the cluster in the plane from different direcion of x-axis and y-axis
    print(CDbw(cl))
    '''

#%%     ######  Step 5  ######

print('\n######  Step 5  ######')

if c > 1:
    CDbwVector = np.zeros(c)
    for cl in clusters:
        cl.objectsList.append(outliersList[0])
        CDbwVector[i] = CDbw(clusters) - CDbwValue
        cl.objectsList.remove(outliersList[0])

    print(CDbwVector)
#clusters[np.maxposition(CDbwVector)]
