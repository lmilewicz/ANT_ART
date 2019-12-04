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
    return np.sum(o_i*o_j)/np.sqrt(np.sum(o_i*o_i)*np.sum(o_j*o_j))

def getMove(vmax, vType):
    if vType == 'Random':
        v = int(vmax/2)
        return [random.randint(-v, v), random.randint(-v, v)]

def printAntColony(AntColony):
    for (i, ant) in enumerate(AntColony):
        print('Ant number: '+str(i)+'. Label: '+ant.label+'. Coord: '+str(ant.dataObject.coord)
        +'. dataObject Label: '+str(ant.dataObject.label))
        
def printObjects(objects):
    '''for (i, oi) in enumerate(objects):
        if oi.ant != None:
            print(str(i)+str(oi.coord)+str(oi.ant.label))'''
    outArray = np.zeros((len(objects), 2))
    for i, o in enumerate(objects):
        outArray[i] = o.coord
    '''plt.scatter(outArray[:,0], outArray[:,1])
    
    plt.title("Objects on plane")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()'''
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.scatter(outArray[:,0], outArray[:,1])
    
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

def convertToArray(clusters):
    U = []
    for cx in clusters:
        oList = cx.objectsList
        outArray = np.zeros((len(oList), 2))
        for i, o in enumerate(oList):
            outArray[i] = o.coord
        U.append(outArray)
    return U