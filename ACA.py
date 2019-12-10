# -*- coding: utf-8 -*-

import numpy as np
import random
import time
import sys

from otherFunctions import (sim, sigmoid, getMove)
from Ant import cluster
                          

def runACA(Mn, ant_number, s, alpha, beta, v, v_max, X, Y, vType, AntColony, objects):
    startX = startZ = time.time()
    iterX = iterY = 1
    diffX = diffY = 0
    divider = 1/(alpha*(1+((v-1)/v_max)))

    #objectsArray1 = np.array(len(objects), 2)
    #objectsArray2 = np.array(len(objects), 4)
    #
    #for i, oi in enumerate(objects):
    #    objectsArray1[i] = oi.coord
    #    objectsArray2[i] = oi.data
        
    for i in range(Mn): # for i = 1, 2, ..., Mn
        for j in range(ant_number): # for j = 1, 2, ..., ant_number
            ant = AntColony[j]
    
            oi = ant.dataObject
            mask_min = np.maximum(0, oi.coord - s)
            mask_max = np.minimum(oi.coord + s, [X, Y])
    
            #outA = objectsArray2[objectsArray1 < mask_max and objectsArray1 > mask_min]
            sim_sum = 0
            startY = time.time()
            
            for oj in objects:
                if np.all(oj.coord > mask_min) and np.all(oj.coord < mask_max) and oi != oj:
                    sim1 = 1-sim(oi.data, oj.data)
                    #sim1 = np.abs(np.linalg.norm(oi.data - oj.data))
                    sim_sum = sim_sum + 1 - sim1*divider
                    #sim_sum = sim_sum + (1-np.linalg.norm(oi.data-oj.data)*divider)
            #localObjects = objectsArray[objectsArray>mask_min and objectsArray<mask_max]
            #for oj in localObjects:
            #    if oi.data != oj[1]:
            #        sim_sum = sim_sum + (1-((1-sim(oi.data, oj.data))*divider))
            
            endY = time.time()
            diffY = diffY+((endY-startY)-diffY)/iterY
            iterY = iterY+1
            #foi = max(0, sim_sum/(s*s))
            foi = sim_sum/(s*s)
            rand = random.random()
            
            
            if ant.label == 'Unloaded':
                picking_prob = 1 - sigmoid(foi, beta)
                if picking_prob > rand and ant.dataObject.antLabel == 'Unloaded':
                    ant.loadObject()
                    ant.move(getMove(v, vType))
                else:
                    ant.setObject(objects[random.randint(0,len(objects)-1)])
            else:
                droping_prob = sigmoid(foi, beta)
                if droping_prob > rand:
                    ant.dropObject()
                    ant.setObject(objects[random.randint(0,len(objects)-1)])
                else:
                    ant.move(getMove(v, vType))
        
        
        endX = time.time()
        diffX = diffX+((endX-startX)-diffX)/iterX
        iterX = iterX+1
        startX = time.time()

        if i%(int(Mn/100)) == 0: 
            sys.stdout.write('\rProgress: %0.2f percent. Execution time: %0.6f. DiffX: %0.6f. Time: %0.1f' % (100*(i+1)/Mn, diffX, diffY*ant_number, endX - startZ))#endX - startX
            sys.stdout.flush()
        
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
'''
def checkClusters(cluster1, cluster2):
    i = 0
    list1 = cluster1.objectsList
    list2 = cluster2.objectsList
    for oi in list1:
        i_prev = i
        for oj in list2:
            if oi == oj:
                i = i + 1
                break
        if i_prev == i and len(list1) < len(list2):
            return []
    return True
'''
def getClustersACA(objects, s, X, Y):
    
    neigh_min = 5
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
                '''if i < c - 1 and checkClustersIfEqual(cl, clusters[c-1]):
                    for oi in clusters[c-1].objectsList:
                        oi.unsetCluster()
                    clusters.pop()
                    c = c - 1
                    break'''
                if i < c - 1 and set(clusters[c-1].objectsList) <= set(cl.objectsList):
                    for oi in clusters[c-1].objectsList:
                        oi.unsetCluster()
                    clusters.pop()
                    c = c - 1
                    break
            clustersCopy = clusters.copy()
            for i, cl in enumerate(clustersCopy):
                if i < c - 1 and set(clusters[c-1].objectsList) >= set(cl.objectsList):
                    for oi in cl.objectsList:
                        oi.unsetCluster()
                    del clusters[i]
                    c = c - 1
                    break
                
    return clusters, outliersList