#!/usr/bin/env python
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import baker

from matrices import *
from spclust import *

@baker.command(default=True)
def basic(blocksize=3,num_blocks=4,strength=0.2,k=None):
    if k is None:
        k=num_blocks
    A = symmetric_blocky_trans(blocksize=blocksize,num_blocks=num_blocks,strength=strength)

    plt.figure()
    plt.subplot(2,2,1)
    plt.gca().matshow(A)
    plt.title('original matrix')

    A_permuted = permute_matrix(A)

    plt.subplot(2,2,2)
    plt.gca().matshow(A_permuted)
    plt.title('permuted matrix')


    _, vecs = get_spectrum(A)

    plt.subplot(2,2,3)
    plt.plot(vecs[:,-2],vecs[:,-3],'k.')

    # passing nclusters=num_blocks is cheating a bit
    recover_perm = find_blockifying_perm(A_permuted,k=k,nclusters=num_blocks)

    plt.subplot(2,2,4)
    plt.gca().matshow(permute_matrix(A_permuted,recover_perm))
    plt.title('recovered blockiness')

    plt.show()

if __name__ == '__main__':
    baker.run()
