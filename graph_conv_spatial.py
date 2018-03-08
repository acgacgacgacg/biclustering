# reference: https://arxiv.org/pdf/1312.6203.pdf p3

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def get_common_neighbors(A, i, j):
    l = A.shape[0]
    ids_i, = np.where(A[i]>0)
    ids_j, = np.where(A[j]>0)
    result = 0
    for x in ids_i:
        for y in ids_j:
            result += A[x, y]
    return result


G = nx.karate_club_graph()
# G = nx.from_numpy_matrix(A)
nx.draw_spectral(G)
plt.show()
A = nx.adjacency_matrix(G)
A = np.array(A.todense())

# parameters for synthtization
n = 50      # size of matrix
size_cluster = 10
number_cluster = n / size_cluster
noise_ratio = 0.01


# relation table R
R = np.zeros((n, n), dtype=np.float)
for i in range(number_cluster):
    R[i*size_cluster:(i+1)*size_cluster, i*size_cluster:(i+1)*size_cluster] = 1

# add some noise
# for i in range(int(n*n*noise_ratio)):
#     x = np.random.randint(0, n)
#     y = np.random.randint(0, n)
#     R[x, y] = 1 - R[x, y]

# apply permutation
# R_ = R[np.random.permutation(n)].T[np.random.permutation(n)].T
R_ = R

# Display R
# fig, axes = plt.subplots(1,1)
# axes.imshow(R_, interpolation='None', cmap='Greys')
# plt.show()


# Construct adjacency matrix
A = np.vstack((np.hstack((np.zeros((50, 50), dtype=np.float), R_)), np.hstack((R_.T, np.zeros((50, 50), dtype=np.float)))))

# Construct neighbor filter matrix according to common neighbors
N = np.zeros(A.shape)
for i in range(A.shape[0]):
    for j in range(A.shape[0]):
        N[i, j] = get_common_neighbors(A, i, j)

N = N / np.sum(N, axis=1)[:, np.newaxis]

sigma = 0.1     # threshold
N = N * (N>=sigma)
# print N




