from __future__ import division
from numpy import *
from matplotlib.pyplot import *

from spclust import *
from matrices import *

f = np.load('/Users/mattjj/Desktop/some_transition_matrices.npz')
mat = f['trans_mat']
A = mat[-1]
A = (A+A.T)/2 # based on experiments, symmetrization IS important for this method!
nonzero_rows, = np.where(A.sum(1) != 0)
nonzero_cols, = np.where(A.sum(0) != 0)
A = A[nonzero_rows][:,nonzero_cols]
A_unnorm = A.copy()
A = symmetric_normalize(A)
perm = find_blockifying_perm(A,k=10,nclusters=10)
matshow(permute_matrix(A_unnorm,perm))

show()
