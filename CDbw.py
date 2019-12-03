# -*- coding: utf-8 -*-

#Clustering Validity Index

import numpy as np
import time

from scipy.spatial import distance


def density(u, U):
    U_stdDev = np.std(U)
    x = distance.cdist([u], U) - U_stdDev
    #for u_i in U:
    #    if np.linalg.norm(u_i - u) <= U_stdDev:
    #        out = out + 1
    return len(x[x<=0])
  
def densityZ(z_ij, U_i, U_j):
    U_ij_stdDev = (np.std(U_i)+np.std(U_j))/2
    '''out = 0
    for u_i in U_i:
        if np.linalg.norm(u_i - z_ij) <= U_ij_stdDev:
            out = out + 1
    for u_j in U_j:
        if np.linalg.norm(u_j - z_ij) <= U_ij_stdDev:
            out = out + 1
    '''
    x_i = distance.cdist([z_ij], U_i) - U_ij_stdDev
    x_j = distance.cdist([z_ij], U_j) - U_ij_stdDev
   
    return (len(x_i[x_i<=0])+len(x_j[x_j<=0]))/(len(U_i)+len(U_j))

def closeRepCal(U_i, U_j):
    dist = distance.cdist(U_i, U_j)
    close_rep_dist = np.min(np.min(dist, axis=1))
    [rep_i_min, rep_j_min] = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    
    return close_rep_dist, rep_i_min, rep_j_min
  
def Intra_den(U):
    out = 0
    for U_i in U:
        for u_ij in U_i:
            out = out + density(u_ij, U_i)
        out = out/len(U_i)
    return out/len(U)

def Inter_den(U):
    out = 0
    for i in range(len(U)):
        for j in range(len(U)):
            if i != j:
                close_rep_dist, rep_i_min, rep_j_min = closeRepCal(U[i], U[j])
                z_ij = [np.round((rep_i_min[0]+rep_j_min[0])/2), np.round((rep_i_min[0]+rep_j_min[0])/2)]
                out = out + close_rep_dist*densityZ(z_ij, U[i], U[j])/(np.std(U[i])+np.std(U[j]))
    return out

def Sep(U):
    out = 0
    for i in range(len(U)):
        for j in range(len(U)):
            if i != j:
                close_rep_dist, x, y = closeRepCal(U[i], U[j])
                out = out + close_rep_dist
    return out

def CDbw(U):
    start1 = time.time()
    Iden = Intra_den(U)
    end1  = time.time()
    SepV = Sep(U)
    end2 = time.time()
    print('Execution time 1: %0.4f. Execution time 2: %0.4f' % (end1 - start1, end2 - end1))
    #print('Iden: %0.2f. SepV: %0.2f' % (Iden, SepV))
    
    return Iden*SepV