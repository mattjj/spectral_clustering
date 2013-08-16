from __future__ import division
import numpy as np

def symmetric_normalize(A,maxiter=5000):
    for i in xrange(maxiter):
        prev = A.copy()
        A /= A.sum(0)
        A /= A.sum(1)[:,None]
        if np.allclose(prev,A,atol=1e-10,rtol=1e-12):
            return A
    print 'warning: reached max iter'
    return A

def permute_matrix(A,perm=None):
    if perm is None:
        perm = np.random.permutation(A.shape[0])
    return A[np.ix_(perm,perm)]

def symmetric_blocky_trans(blocksize,num_blocks,p_on=0.2,strength=0.2):
    n = blocksize*num_blocks
    out = np.zeros((n,n))

    bd_support = np.kron(np.eye(num_blocks,dtype=bool),np.ones((blocksize,blocksize),dtype=bool))
    out[bd_support] = np.random.uniform(0.5,1,size=bd_support.sum())

    obd_support = np.logical_and(
            np.logical_not(bd_support),
            np.random.uniform(size=(n,n)) < p_on)
    out[obd_support] = np.random.uniform(0,strength,size=obd_support.sum())

    out = (out+out.T)/2
    out = symmetric_normalize(out)

    return out


# TODO how blocky do non-blocky random matrices look when asked to blockify?
# TODO independent runs look totally different
# TODO kmeans may suck
