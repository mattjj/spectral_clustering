from __future__ import division
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from matplotlib import pyplot as plt


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

        if self.n_blocks is not None:
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
        print('warning: reached max iter')
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

