# -*- coding: utf-8 -*-

import numpy as np

class Ant:
    def __init__(self, dataObject):
        self.dataObject = dataObject
        self.label = 'Unloaded'
        self.cluster = 0
    def setObject(self, dataObject):
        self.dataObject = dataObject
    def loadObject(self):
        self.label = 'Loaded'
        self.dataObject.antLabel = 'Loaded'
    def dropObject(self):
        self.dataObject.antLabel = 'Unloaded'
    def moveAnt(self):
        self.dataObject.coord = self.dataObject.coord + [1, -1] ## TO DO


class dataObject:
    def __init__(self, coord):
       self.coord = np.array(coord)
       self.antLabel = 'Unloaded'