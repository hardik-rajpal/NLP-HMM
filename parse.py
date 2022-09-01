import nltk
import numpy as np
from nltk.corpus import brown
sentences = [sent for sent in brown.tagged_sents()[0:2]]


# Tm = (ij) -> transition from state s_i to s_j
# Em = (ij) -> observing o_j from s_i state
# pi = (i) -> initial probab that x1 = s_i
# B = (ij) -> probab that Y_j will be in state s_i
# A = (ij) -> probab that s_j will be after s_i

K = 10 # no of states
N = 10 # no of observations of Y

A = np.zeros((K, K))
B = np.zeros((K, N))
Y = np.zeros((N))
pi = np.zeros((K))
S = np.zeros((K, 1))
O = np.zeros((N, 1))

def viterbi(O, S, pi, Y, A, B):
    T1 = np.zeros((K, N))
    T2 = np.zeros((K, N))

    inter = pi * B[:, int(Y[0])]
    T1[:, 0] = inter

    for obs in range(1, N):
        B_factor = B[:, int(Y[obs])].reshape(1,K)
        B_matrix_factor = np.tile(B_factor, (K,1))
    
        A_factor = A * B_matrix_factor
        final = T1[:, obs-1] * A_factor
    
        T1[:, obs] = np.max(final, 0)
        T2[:, obs] = np.argmax(final, 0)
    
    point = np.argmax(T1[:, N-1])
    best_path = []

    for obs in range(N-2, -1, -1):
        best_path.append(S[point])
        point = int(T2[point, obs])

    return best_path

viterbi(O, S, pi, Y, A, B)

    # for state in range(K):
    #     T1[state, 0] = pi[state] * B[state, Y[0]]
    
    # for state in range(K):
    #     T1[state, obs] = np.max(inter)
    #     T2[state, obs] = np.argmax(inter)