# TODO
# Try out AffinityPropagation
# Raise errors for n_eigenvectors
# Raise error for n_blocks


from __future__ import division
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


"""import numpy as np
import pandas as pd
import pymouse
import os
import spectral_clustering.spclust as sp
from numpy import newaxis as na
from pymouse import entropy

def norm_statemap(A):
    d = np.repeat(A.sum(1)[:,na], len(A), 1)
    d[d==0] = 1.0
    A = A/d
    return A

import numpy as np
import pandas as pd
import pymouse
import os
import spectral_clustering.spclust as sp
from numpy import newaxis as na
from pymouse import entropy

def norm_statemap(A):
    d = np.repeat(A.sum(1)[:,na], len(A), 1)
    d[d==0] = 1.0
    A = A/d
    return A

big_df = df[(df.n_states == 200) & (df.libsize == 200)].reset_index().iloc[1]
small_df = best_df.copy()
this_df = big_df.copy()

A = this_df['transition_matrix']
Ad = this_df['deployment_matrix']
A = pymouse.entropy.crop_trans_matrix(A)
Ad = pymouse.entropy.crop_trans_matrix(Ad)

H = np.zeros_like(A)
n = H.shape[0]
for j in range(n):
    not_j = np.arange(n)!=j
    d = (np.eye(n-1) - A[np.ix_(not_j,not_j)])
    di = np.linalg.inv(d)
    H[np.ix_(not_j,[j])] = np.dot(di,np.ones((n-1,1)))

n_blocks = 7
n_eigvecs = 7
figsize = (10,10)


# Show the permuted transition matrices
newBlock = sp.SpectralBlockify(n_blocks,n_eigvecs)
newBlock.fit(H)
q = newBlock.permute(Ad)
figure()
gs = GridSpec(1,2, hspace=0.1, wspace=0.01, width_ratios=[100,5])
figure(figsize=figsize)
subplot(gs[0])
imshow(q)
subplot(gs[1])
imshow(newBlock.block_labels_[:,na][newBlock.permutation_], cmap='Paired')
axis('off')

# Show the permuted flow matrices
Hn = newBlock._symmetric_normalize(H)
for i in range(len(Hn)):
    Hn[i,i] = np.nan
q = newBlock.permute(np.sqrt(1.0/Hn))
gs = GridSpec(1,2, hspace=0.1, wspace=0.01, width_ratios=[100,5])
figure(figsize=figsize)
subplot(gs[0])
imshow(q)
subplot(gs[1])
imshow(newBlock.block_labels_[:,na][newBlock.permutation_], cmap='Paired')
axis('off')


# Show the permuted transition matrices
newBlock = sp.SpectralBlockify(n_blocks,n_eigvecs)
newBlock.fit(Ad)
q = newBlock.permute(Ad)
gs = GridSpec(1,2, hspace=0.1, wspace=0.01, width_ratios=[100,5])
figure(figsize=figsize)
subplot(gs[0])
imshow(q)
subplot(gs[1])
imshow(newBlock.block_labels_[:,na][newBlock.permutation_], cmap='Paired')
axis('off')
"""

class SpectralBlockify(object):
    """Find blocks in a matrix"""
    def __init__(self, n_blocks=None, n_eigenvectors=None, order_within_blocks=True, order_between_blocks=True, ap_damping=0.5, ap_preference=None):
        super(SpectralBlockify, self).__init__()
        self.n_blocks = n_blocks
        self.n_eigenvectors = n_eigenvectors
        self.order_within_blocks = order_within_blocks
        self.order_between_blocks = order_between_blocks
        self.ap_damping = ap_damping
        self.ap_preference = ap_preference

    def fit(self, A):
        """Find the blocks within A
        """
        # First, symmetricize and normalize A
        this_A = self._symmetric_normalize(A)
        # Then, get the spectrum (we'll cluster based on the matrix's eigenvectors)
        _, vecs = self._get_spectrum(this_A)
        # Get the 2nd to 2+n_eigenvectors eigenvectors to cluster
        # (we ignore the first eigenvector, that's the matrix's steady-state)
        eigvecs_to_cluster = vecs[:,-(self.n_eigenvectors+1):-1]
        # Then, cluster the eigenvectors

        if self.n_blocks != None:
            self.block_labels_ = KMeans(self.n_blocks).fit_predict(eigvecs_to_cluster)
        else:
            apNode = AffinityPropagation(damping=self.ap_damping, preference=self.ap_preference).fit(A)
            self.block_labels_ = apNode.labels_

        if self.order_between_blocks:
            # Get the within-block mean entropy
            new_order = np.zeros_like(np.unique(self.block_labels_))
            mean_probs = [A[self.block_labels_ == i_block].mean() for i_block in np.unique(self.block_labels_)]
            new_order = np.argsort(mean_probs)[::-1]

            # Swap the labels
            self.block_labels_ = new_order[self.block_labels_]
            

        # With the clustering, create the permutation
        if self.order_within_blocks == False:
            self.permutation_ = np.argsort(self.block_labels_)
        else:
            within_block_idx = []
            this_permutation = []
            
            for i_block in np.unique(self.block_labels_):
                # Get all of the out-of-block values
                # For each row within the block
                block_idx = self.block_labels_ == i_block
                non_block_idx = self.block_labels_ != i_block
                out_of_block_vals = this_A[np.ix_(block_idx, non_block_idx)]
                # Find the summed probability out-of-block
                out_of_block_probability = np.abs(out_of_block_vals).sum(1)
                within_block_idx.append(np.argsort(out_of_block_probability))
                # Now, create the permutation
                this_permutation.append(np.argwhere(block_idx)[np.argsort(out_of_block_probability)].ravel())

            # Now create a permutation
            self.permutation_ = np.hstack(this_permutation)

        return self

    def permute(self, A):
        return A[np.ix_(self.permutation_, self.permutation_)]

    def fit_permute(self, A):
        self.fit(A)
        return self.permute(A)


    def _get_spectrum(self, A):
        vals, vecs = np.linalg.eig(A)
        order = np.argsort(np.abs(vals))
        return vals[order], vecs[:,order]

    def _symmetric_normalize(self, A,maxiter=500):
        A = (A+A.T)/2.0
        for i in xrange(maxiter):
            prev = A.copy()
            d = A.sum(0)
            d[d==0] = 1.0
            A /= d
            d = A.sum(1)[:,None]
            d[d==0] = 1.0
            A /= d
            if np.allclose(prev,A,atol=1e-10,rtol=1e-12):
                return A
        print 'warning: reached max iter'
        return A

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

