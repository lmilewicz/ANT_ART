# -*- coding: utf-8 -*-

#Clustering Validity Index

import numpy as np

def convertToArray(objects):
    outArray = np.zeros((len(objects), 2))
    for i, o in enumerate(objects):
        outArray[i] = o.coord
    return outArray

def density(u, Ux):
    out = 0
    U = convertToArray(Ux)
    U_stdDev = np.std(U)
    for u_i in U:
        if np.linalg.norm(u_i - u) <= U_stdDev:
            out = out + 1
    return out
  
def densityZ(z_ij, U_i, U_j):
    out = 0
    U_ij_stdDev = (np.std(U_i)+np.std(U_j))/2
    for u_i in U_i:
        if np.linalg.norm(u_i - z_ij) <= U_ij_stdDev:
            out = out + 1
    for u_j in U_j:
        if np.linalg.norm(u_j - z_ij) <= U_ij_stdDev:
            out = out + 1
    return out/(len(U_i)+len(U_j))

def closeRepCal(U_ix, U_jx):
    close_rep_dist = 99999
    U_i = convertToArray(U_ix)
    U_j = convertToArray(U_jx)
    rep_i_min = rep_j_min = [0, 0]
    for rep_i in U_i:
        for rep_j in U_j:
            temp = np.linalg.norm(rep_i-rep_j)
            if close_rep_dist > temp:
                close_rep_dist = temp
                rep_i_min = rep_i
                rep_j_min = rep_j
    return close_rep_dist, rep_i_min, rep_j_min
  
def Intra_den(U):
    out = 0
    for U_i in U:
        U_i = U_i.objectsList
        for u_ij in U_i:
            out = out + density(u_ij.coord, U_i)
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
                close_rep_dist, x, y = closeRepCal(U[i].objectsList, U[j].objectsList)
                out = out + close_rep_dist
    return out

def CDbw(U):
    return Intra_den(U)*Sep(U)