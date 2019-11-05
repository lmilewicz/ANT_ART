# -*- coding: utf-8 -*-


class Ant:
    def __init__(self, coord):
       self.label = 'Unloaded'
       self.coord = coord
    def setLabel(self, label):
        self.label = label
    def setCoord(self, coord):
        self.coord = coord

