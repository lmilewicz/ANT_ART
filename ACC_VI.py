# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import sys

from CDbw import CDbw
from Ant import Ant
from ACA import (runACA, getClustersACA)
from otherFunctions import (printObjects, printClusters, returnObjects, convertToArray)


#%%     ######  Step 0  ######

# Maximum number of ants:
ant_number = 50

# Maximum number of iterations:
Mn = 100000

# Side length of local region:
s = 20

# Maximum speed of ants moving
v_max = 30

# Alpha & beta parameters
alpha = 0.04
beta = 0.3

# Quantity of objects
N = 150

# x dimension
X = 100

# y dimension
Y = 100

# create a space
space = np.zeros((X, Y))


# declare v
v = 30


#%%
#Project the data on the plane - give a pair of coordinate (x,y) to each object randomly
#Each ant that is currently unloeaded chooses an object at random

print('######  Step 1  ######')

start = time.time()

objects = returnObjects(X, Y, N, 'iris')

AntColony = []
for i in range(ant_number):
    AntColony.append(Ant(objects[random.randint(0,len(objects)-1)]))

printObjects(objects, 'coord')
#printObjects(objects, 'data')


end = time.time()
print('Execution time: %0.2f' % (end - start))

#%%
print('\n######  Step 2  ######')

start = time.time()

vType = 'Random'
runACA(Mn, ant_number, s, alpha, beta, v, v_max, X, Y, vType, AntColony, objects)
printObjects(objects, 'coord')
printObjects(objects, 'data')

end = time.time()
print('\n\nExecution time: %0.2f' % (end - start))

#%%
print('\n######  Step 3  ######')

start = time.time()

clusters, outliersList = getClustersACA(objects, s, X, Y)
c = len(clusters)

end = time.time()
print('Execution time: %0.2f' % (end - start))

#%%
print('\n######  Step 4  ######')

start = startX = time.time()

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

#%%

printClusters(clusters, 'data')
printClusters(clusters, 'coord')

#clusters[np.maxposition(CDbwVector)]
