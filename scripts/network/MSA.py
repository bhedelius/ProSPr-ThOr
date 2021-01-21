import string
import numpy as np
from scipy.spatial.distance import pdist, squareform

from Bio import Alphabet
from Bio.Alphabet import Reduced

import random

# %%
class MSA:
    ''' Copies trRosetta's methods'''
    def __init__(self, filename, reduced=Alphabet.Reduced.murphy_10_tab, full=True):
        # Prepared reduced alphabet
        self.reduced = reduced
        self.red_chars = self.get_red_chars(self.reduced)

        # Parse sequence alignment into various formats
        seqs = self.parse(filename)
        if full is False:
            # Shuffle all but initial
            shuf = seqs[1:]
            random.shuffle(shuf)
            # Take either half or 500 sequences, depending on which is smaller
            shuf = shuf[:len(shuf)//2]
            shuf = shuf[:500]
            seqs = [seqs[0]]
            # Add initial sequence back
            shuf.insert(0,seqs[0])
            seqs = shuf
        self.seqs = seqs
        self.ints = self.get_ints(self.seqs)
        self.hot = self.one_hot(self.ints)

        self.red_seqs = self.reduce(self.seqs, self.reduced)
        self.red_ints = self.get_ints(self.red_seqs, self.red_chars)
        self.red_hot = self.one_hot(self.red_ints,10)

        if len(seqs)==1:
            return

        # Calculate Parameters
        self.weights = self.calc_weights(self.ints)
        self.freq = self.calc_freq(self.hot, self.weights)
        self.dca = self.calc_dca(self.hot, self.weights)
        self.apc = self.calc_apc(self.dca)
        self.red_dca = self.calc_dca(self.red_hot, self.weights)

    @staticmethod
    def parse(filename):
        seqs = []
        table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
        for line in open(filename, "r"):
            if line[0] != '>':
                seqs.append(line.rstrip().translate(table))
        return seqs

    @staticmethod
    def reduce(seqs, reduced=Alphabet.Reduced.murphy_10_tab):
        red_seqs = []
        table = str.maketrans(reduced)
        for seq in seqs:
            red_seqs.append(seq.translate(table))
        return red_seqs

    @staticmethod
    def get_red_chars(reduced):
        values = list(set(reduced.values()))
        values.sort()
        return "".join(values)

    @staticmethod
    def get_ints(seqs, chars="ACDEFGHIKLMNPQRSTVWY"):
        alphabet = np.array(list(chars), dtype='|S1').view(np.uint8)
        ints = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            ints[ints == alphabet[i]] = i
        ints[ints > len(chars)] = len(chars)
        return ints

    @staticmethod
    def one_hot(ints, q=20):
        return (np.arange(q+1) == ints[..., None]).astype(int)[:, :, :-1]
    @staticmethod
    def calc_weights(hot, cutoff=0.8):
        return 1. / (1 + np.sum(squareform(pdist(hot, "hamming") <= cutoff), axis=1))

    @staticmethod
    def calc_freq(hot, weights):
        return np.tensordot(weights, hot, axes=1) / np.sum(weights)

    # weighted covariance
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/weigcorr.htm
    @staticmethod
    def calc_dca(hot, weights):
        (B, N, q) = hot.shape
        B_eff = np.sum(weights)
        x = np.reshape(hot, (B, N * q))
        mean = np.sum(x * weights[:, None], axis=0) / B_eff
        x = (x - mean) * np.sqrt(weights[:, None])
        cov = np.matmul(np.transpose(x), x) / B_eff
        factor = np.sqrt(21)/np.sqrt(np.sum(weights))
        inv_cov = np.linalg.inv(cov + factor*np.eye(N * q)) - np.eye(N * q)/factor  # Moore-Penrose inverse
        J = np.reshape(inv_cov, (N, q, N, q))
        return np.transpose(J, (0, 2, 1, 3))

    @staticmethod
    def calc_apc(dca):
        norm = np.linalg.norm(dca, axis=(2, 3))
        return norm - np.sum(norm, axis=0, keepdims=True) * np.sum(norm, axis=1, keepdims=True) / np.sum(norm)

    def hodca(self, il, ih, jl, jh, kl, kh):
        R = self.red_hot - np.sum(self.red_hot, axis=0, keepdims=True)
        S = np.tensordot(R, self.red_dca, axes=((1, 2), (1, 3)))
        S *= np.cbrt(self.weights[:, None, None])
        V = np.tensordot(S[:, il:ih, :, None, None] * S[:, None, None, jl:jh, :], S[:, kl:kh, :], axes=(0, 0))
        V = np.transpose(V,(1,3,5,0,2,4))
        return np.reshape(V,(1000,ih-il,jh-jl,kh-kl))


#msa = MSA('./data/3k0bA02a')
#import matplotlib.pyplot as plt
#plt.imshow(msa.apc*(1-np.eye(55)))
#plt.show()

#V = msa.hodca(0,16,0,16,0,16)
#Vp = np.linalg.norm(V[3],axis=3)
#Vp = msa.calc_apc(Vp)
#plt.imshow(Vp)
#plt.show()
