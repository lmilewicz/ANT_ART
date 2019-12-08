# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import sys

from otherFunctions import (sim, sigmoid, getMove, printObjects)
from Ant import cluster
                          

def runACA(Mn, ant_number, s, alpha, beta, v, v_max, X, Y, vType, AntColony, objects):
    startX = time.time()
    
    for i in range(Mn): # for i = 1, 2, ..., Mn
        for j in range(ant_number): # for j = 1, 2, ..., ant_number
            ant = AntColony[j]
    
            oi = ant.dataObject
            mask_min = np.maximum(0, oi.coord - s)
            mask_max = np.minimum(oi.coord + s, [X, Y])
    
            sim_sum = 0
            for oj in objects:
                if oi != oj and np.all(oj.coord > mask_min) and np.all(oj.coord < mask_max):
                    sim_sum = sim_sum + (1-(1-sim(oi.data, oj.data)/(alpha*(1+((v-1)/v_max)))))
    
            foi = max(0, sim_sum/(s*s))
            rand = random.random()
    
            if ant.label == 'Unloaded':
                picking_prob = 1 - sigmoid(foi, beta)
                if picking_prob > rand and ant.dataObject.antLabel == 'Unloaded':
                    ant.loadObject()
                    ant.move(getMove(v_max, vType))
                else:
                    ant.setObject(objects[random.randint(0,len(objects)-1)])
            else:
                droping_prob = sigmoid(foi, beta)
                if droping_prob > rand:
                    ant.dropObject()
                    ant.setObject(objects[random.randint(0,len(objects)-1)])
                else:
                    ant.move(getMove(v_max, vType))
        
        if i%(int(Mn/100)) == 0: 
            endX = time.time()
            sys.stdout.write('\rProgress: %0.2f percent. Execution time: %0.2f' % (100*(i+1)/Mn, endX - startX))
            sys.stdout.flush()
            startX = time.time()

    printObjects(objects)


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

def getClustersACA(objects, s, X, Y):
    
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
    return clusters, outliersList