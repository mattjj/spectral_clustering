from __future__ import division
import numpy as np


def symmetric_normalize(A,maxiter=5000):
    for i in range(maxiter):
        prev = A.copy()
        A /= A.sum(0)
        A /= A.sum(1)[:,None]
        if np.allclose(prev,A,atol=1e-10,rtol=1e-12):
            return A
    print('warning: reached max iter')
    return A


def permute_matrix(A,perm=None):
    if perm is None:
        perm = np.random.permutation(A.shape[0])
    return A[np.ix_(perm,perm)]


def symmetric_blocky_trans(blocksize,num_blocks,p_on=0.2,strength=0.2):
    n = blocksize*num_blocks
    out = np.zeros((n,n))

    bd_support = np.kron(
        np.eye(num_blocks,dtype=bool),
        np.ones((blocksize,blocksize),dtype=bool))
    out[bd_support] = np.random.uniform(0.5,1,size=bd_support.sum())

    obd_support = np.logical_and(
            np.logical_not(bd_support),
            np.random.uniform(size=(n,n)) < p_on)
    out[obd_support] = np.random.uniform(0,strength,size=obd_support.sum())

    out = (out+out.T)/2
    out = symmetric_normalize(out)

    return out


def asymmetric_blocky_trans(blocksize,num_blocks,p_on=0.2,strength=0.2):
    n = blocksize*num_blocks
    out = np.zeros((n,n))

    bd_support = np.kron(
            np.eye(num_blocks,dtype=bool),
            np.ones((blocksize,blocksize),dtype=bool))
    out[bd_support] = np.random.uniform(0.5,1,size=bd_support.sum())

    obd_support = np.logical_and(
        np.logical_not(bd_support),
        np.random.uniform(size=(n,n)) < p_on)
    out[obd_support] = np.random.uniform(0,strength,size=obd_support.sum())

    out /= out.sum(1)[:,None]

    return out


def get_spectrum(A):
    vals, vecs = np.linalg.eig(A)
    order = np.argsort(np.abs(vals))
    return vals[order], vecs[:,order]


# defined in http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
def graph_laplacian(A):
    D = np.diag(A.sum(1))
    return D - A


def sym_normalized_graph_laplacian(A):
    L = graph_laplacian(A)
    d = A.sum(1)
    return (L / np.sqrt(d)) / np.sqrt(d)[:,None]


def rw_normalized_graph_laplacian(A):
    L = graph_laplacian(A)
    d = A.sum(1)
    return L / d[:,None]


def bandify(A,laplacian=graph_laplacian):
    _, vecs = get_spectrum(laplacian(A))
    perm = np.argsort(vecs[:,1])
    return permute_matrix(A,perm)


# TODO kmeans may suck. kmeans++ init!
