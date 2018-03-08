# reference: https://tkipf.github.io/graph-convolutional-networks/

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import expit

# np.random.seed(1)

G = nx.karate_club_graph()
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

D = np.diag(np.sum(A, axis=1)**-0.5)
DAD = np.dot(np.dot(D, (A + np.identity(A.shape[0], dtype=np.float))), D)


# propagation
W = np.random.rand(100, 100) * 2 - 1
H = np.tanh(np.dot(DAD, W))
H = H / np.sum(H, axis=1)[:, np.newaxis]

for i in range(20):
    W = np.random.rand(100, 100) * 2 - 1    # initialize weight
    H = np.tanh(np.dot(np.dot(DAD, H), W))
    H = H / np.sum(H, axis=1)[:, np.newaxis]

W = np.random.rand(100, 2) * 2 - 1
H = np.tanh(np.dot(np.dot(DAD, H), W))

# print DAD
fig, ax = plt.subplots()
for color, i in zip(['red', 'green', 'blue', 'grey', 'brown'], range(number_cluster)):
    x, y = H[i*size_cluster:(i+1)*size_cluster,0], H[i*size_cluster:(i+1)*size_cluster,1]
    ax.scatter(x, y, c=color, label=color, edgecolors='none')

# ax.legend()
ax.grid(True)




# plt.scatter(H4[:,0], H4[:,1], marker='x')
plt.show()