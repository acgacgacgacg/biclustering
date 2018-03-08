import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def is_sim(R, i, j, k):
    # given node i, return if j is i's k-cn node
    # if np.dot(R[i], R[j]) >= k:
    #     return True
    # else:
    #     return False

    # Jaccard
    R_i = np.sum(R[i], axis=0) >= 1
    R_j = np.sum(R[j], axis=0) >= 1

    if np.sum((R_i+R_j)>=1) != 0:
        J = float(np.sum(R_i*R_j)) / float(np.sum((R_i+R_j)>=1))
    else:
        J = 0
    return J >= k



# parameters for synthtization
N = 100      # size of matrix
K = 0.6       # param for sim
size_cluster = 10
number_cluster = N / size_cluster
noise_ratio = 0.05


# relation table R
R = np.zeros((N, N), dtype=np.float)

# diagonal speratable clusters
# for i in range(number_cluster):
#     R[i*size_cluster:(i+1)*size_cluster, i*size_cluster:(i+1)*size_cluster] = 1

np.random.seed(3)

# random clusters
for i, j in np.random.randint(0, N-size_cluster, size=(number_cluster, 2)):
    R[i:i+size_cluster, j:j+size_cluster] = 1




# add some noise
for i in range(int(N*N*noise_ratio)):
    x = np.random.randint(0, N)
    y = np.random.randint(0, N)
    R[x, y] = 1 - R[x, y]

# apply permutation
# R_ = R[np.random.permutation(N)].T[np.random.permutation(N)].T
R_ = R




def propagate(l_set_nodes, R):
    n = len(l_set_nodes)
    lv = []
    # propagate for row
    for i in range(n):
        s_i = l_set_nodes[i].copy()
        for j in range(i+1, n):
            if is_sim(R_, list(l_set_nodes[i]), list(l_set_nodes[j]), K):
                for x in l_set_nodes[j]:
                    s_i.add(x)
        flg = True
        for s in lv:
            if s_i <=s:
                flg = False
                break
        if flg:
            lv.append(s_i)
    return lv

lvs = []
ini_l_set_nodes = [set([i]) for i in range(N)]
l_set_nodes = ini_l_set_nodes

# propagate 6 times
for t in range(6):
    l_set_nodes = propagate(l_set_nodes, R_)
    lvs.append(l_set_nodes)


for i in range(6):
    print('Level: ' + str(i), len(lvs[i]))
    for x in lvs[i]:
        if len(x) >1:
            print(x)



# Display R
fig, axes = plt.subplots(1,1)
axes.imshow(R_, interpolation='nearest', cmap='Greys')
plt.show()


