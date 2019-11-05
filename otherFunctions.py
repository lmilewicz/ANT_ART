# -*- coding: utf-8 -*-

import math
import numpy as np

def sigmoid(x, beta):
    return 1 / (1+ math.exp(-x * beta))

def sim(o_i, o_j):
    return np.sum(o_i*o_j)/np.sqrt(np.sum(o_i*o_i)*np.sum(o_j*o_j))