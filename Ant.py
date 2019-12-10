# -*- coding: utf-8 -*-

import numpy as np
import random


class Ant:
    def __init__(self, dataObject):
        self.dataObject = dataObject
        self.label = 'Unloaded'
        #self.cluster = 0
    def setObject(self, dataObject):
        self.dataObject = dataObject
    def loadObject(self):
        self.label = 'Loaded'
        self.dataObject.antLabel = 'Loaded'
        self.dataObject.ant = self
    def dropObject(self):
        self.label = 'Unloaded'
        self.dataObject.antLabel = 'Unloaded'
        self.dataObject.ant = None
    def move(self, speed):
        self.dataObject.coord = self.dataObject.coord + speed
        self.dataObject.coord[self.dataObject.coord < 0] = 0
        if self.dataObject.coord[0] > self.dataObject.X:
            self.dataObject.coord[0] = self.dataObject.X-1
        if self.dataObject.coord[1] > self.dataObject.Y:
            self.dataObject.coord[1] = self.dataObject.Y-1
            

class dataObject:
    def __init__(self, data, X, Y):
       self.data = np.array(data)
       self.X = X
       self.Y = Y
       self.coord = np.array([random.randint(0, X-1), random.randint(0, Y-1)])
       self.clusterList = []
       self.antLabel = 'Unloaded'
       self.label = 'Unclassified'
       self.ant = None
    def setOutlier(self):
       self.label = 'Outlier'
       self.clusterList = []
       self.ant = None
    def unsetCluster(self):
       self.clusterList.pop()
       if len(self.clusterList) == 0:
           dataObject.label = 'Classified'


class cluster:
    def __init__(self, name):
       self.name = name
       self.objectsList = []
    def addObject(self, dataObject):
       self.objectsList.append(dataObject)
       dataObject.label = 'Classified'
       dataObject.clusterList.append(self.name)

    def __repr__(self):
        return '{0}[{1}]'.format(self.name, len(self.objectsList))

    def __str__(self):
        return repr(self)
