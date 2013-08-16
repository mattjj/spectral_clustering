from __future__ import division
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def get_spectrum(A):
    vals, vecs = np.linalg.eig(A)
    order = np.argsort(np.abs(vals))
    return vals[order], vecs[:,order]

def find_blockifying_perm(A,k,nclusters):
    _, vecs = get_spectrum(A)
    return np.argsort(KMeans(n_clusters=nclusters).fit(vecs[:,-(k+1):-1]).labels_)

def plot(A,tol=1e-5,plot=True):
    _, vecs = get_spectrum(A)
    xs, ys = vecs[:,-3], vecs[:,-2]
    if plot:
        plt.figure()
        themax = A.max()
        for i,j in zip(*np.triu_indices_from(A,k=1)):
            if A[i,j] > tol:
                plt.plot((xs[i],xs[j]),(ys[i],ys[j]),'b-',alpha=A[i,j]/themax)
    return xs, ys

def plot3D(A,tol=1e-5,plot=True):
    _, vecs = get_spectrum(A)
    xs, ys, zs = vecs[:,-4], vecs[:,-3], vecs[:,-2]
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        themax = A.max()
        for i,j in zip(*np.triu_indices_from(A,k=1)):
            if A[i,j] > tol:
                ax.plot(np.r_[xs[i],xs[j]],np.r_[ys[i],ys[j]],np.r_[zs[i],zs[j]],
                        'b-',alpha=A[i,j]/themax)
    return xs, ys, zs

