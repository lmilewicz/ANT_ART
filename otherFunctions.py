# -*- coding: utf-8 -*-

import math
import numpy as np
import random
    
import matplotlib.pyplot as plt

from sklearn import datasets

from Ant import dataObject


def sigmoid(x, beta):
    return 1 / (1+ math.exp(-x * beta))

def sim(o_i, o_j):
    return np.sum(o_i * o_j)/np.sqrt(np.sum(o_i ** 2)*np.sum(o_j ** 2))

def getMove(vmax, vType):
    if vType == 'Random':
        v = int(vmax/2)
        return [random.randint(-v, v), random.randint(-v, v)]

def printAntColony(AntColony):
    for (i, ant) in enumerate(AntColony):
        print('Ant number: '+str(i)+'. Label: '+ant.label+'. Coord: '+str(ant.dataObject.coord)
        +'. dataObject Label: '+str(ant.dataObject.label))
        
def printObjects(objects, printType = 'data'):
    outArray = np.zeros((len(objects), 2))
    if printType == 'data':
        for i, o in enumerate(objects):
            outArray[i] = o.data[0,:2]
    else:
       for i, o in enumerate(objects):
            outArray[i] = o.coord
            
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(outArray[:,0], outArray[:,1])
    
    plt.title('Objects on plane')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
   
def printClusters(clusters, printType = 'data'):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    
    color = []
    n = len(clusters)
    for i in range(n):
        color.append('#%06X' % random.randint(0, 0xFFFFFF))
    
    for x, cl in enumerate(clusters):
        outArray = np.zeros((len(cl.objectsList), 2))
        for i, o in enumerate(cl.objectsList):
            if printType == 'data':
                outArray[i] = o.data[0,:2]
            else:
                outArray[i] = o.coord
        ax.scatter(outArray[:,0], outArray[:,1], color = color[x])
       
    plt.title('Objects on plane')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
         
def returnObjects(X, Y, N, dataType):
    objects = []

    if dataType == 'random':
        for i in range(N):
            x = random.randint(0,X-1)
            y = random.randint(0,Y-1)
            objects.append(dataObject([x, y], X, Y))
    elif dataType == 'iris':
        iris = datasets.load_iris()
        if N > len(iris.data):
            N = len(iris.data)
        db_iris = iris.data[:N,:]
        for i in range(N):
            objects.append(dataObject([db_iris[i, :]], X, Y))
    return objects

def convertToArray(clusters):
    U = []
    for cx in clusters:
        oList = cx.objectsList
        outArray = np.zeros((len(oList), 2))
        for i, o in enumerate(oList):
            outArray[i] = o.coord
        U.append(outArray)
    return U