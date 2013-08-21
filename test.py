#!/usr/bin/env python
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import baker

from matrices import *
from spclust import *

@baker.command(default=True)
def basic(blocksize=3,num_blocks=4,strength=0.2,k=None,asymmetric=False):
    if k is None:
        k=num_blocks
    else:
        k = int(k)

    if asymmetric:
        A = asymmetric_blocky_trans(
                blocksize=blocksize,
                num_blocks=num_blocks,
                strength=strength)
    else:
        A = symmetric_blocky_trans(
                blocksize=blocksize,
                num_blocks=num_blocks,
                strength=strength)

    plt.figure()
    gs = GridSpec(6,2,hspace=0.5,wspace=0.5)

    plt.subplot(gs[:3,0])
    plt.gca().matshow(A)
    plt.title('original matrix')

    A_permuted = permute_matrix(A)

    plt.subplot(gs[:3,1])
    plt.gca().matshow(A_permuted)
    plt.title('permuted matrix')

    A_permuted_sym = (A_permuted+A_permuted.T)/2


    vals, vecs = get_spectrum(A_permuted)
    plt.subplot(gs[3,0])
    plt.plot(np.real(vals[::-1]),label='real')
    plt.plot(np.imag(vals[::-1]),label='imag')
    plt.legend()
    plt.title('asymmetric spectrum')

    vals, vecs = get_spectrum(A_permuted_sym)
    plt.subplot(gs[4,0])
    plt.plot(vals[::-1])
    plt.title('symmetric spectrum')
    plt.subplot(gs[5,0])
    plt.plot(vecs[:,-2],vecs[:,-3],'k.')

    # passing nclusters=num_blocks is cheating a bit
    recover_perm = find_blockifying_perm(A_permuted_sym,k=k,nclusters=num_blocks)

    plt.subplot(gs[3:,1])
    plt.gca().matshow(permute_matrix(A_permuted_sym,recover_perm))
    plt.title('recovered blockiness')

    plt.show()

if __name__ == '__main__':
    baker.run()
