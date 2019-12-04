import numpy as np

def similarity(o1, o2):
    return np.dot(o1, o2) / np.sqrt(np.sum(o1**2)*np.sum(o2**2))

def cosine_distance(o1, o2):
    return 1 - similarity(o1, o2)
