# -*- coding: utf-8 -*-

import math
import numpy as np
import random


def sigmoid(x, beta):
    return 1 / (1+ math.exp(-x * beta))

def sim(o_i, o_j):
    return np.sum(o_i*o_j)/np.sqrt(np.sum(o_i*o_i)*np.sum(o_j*o_j))

def getMove():
    return [random.randint(-5, 5), random.randint(-5, 5)]

def printAntColony(AntColony):
    for (i, ant) in enumerate(AntColony):
        print('Ant number: '+str(i)+'. Label: '+ant.label+'. Coord: '+str(ant.dataObject.coord)
        +'. dataObject Label: '+str(ant.dataObject.label))
        
def printObjects(objects):
    for (i, oi) in enumerate(objects):
        if oi.ant != None:
            print(str(i)+str(oi.coord)+str(oi.ant.label))