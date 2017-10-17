import numpy as np 
from sklearn.cluster import KMeans


# Input A: a relational matrix
# Output B: clustered ralational matrix of A (k=2)
def spectral_bicluster(A):
	m, n = A.shape
	D1 = np.diag(np.sum(A, axis=1)**(-0.5))
	D2 = np.diag(np.sum(A, axis=0)**(-0.5))
	An = np.dot(np.dot(D1, A), D2)
	U, s, V = np.linalg.svd(An, full_matrices=True)
	# print 'u1=', U[:,0]
	# print 'u2=', U[:,0]
	# print s
	z2 = np.hstack((np.dot(D1, U[:,1]), np.dot(D2, V.T[:, 1]))).reshape(m+n, 1)
	# print z2
	objKmeans = KMeans(n_clusters = 2).fit(z2)
	mu, labels = objKmeans.cluster_centers_, objKmeans.labels_
	idx_d_1 = np.where(labels[:m]==0)[0]
	a = len(idx_d_1)
	idx_d_2 = np.where(labels[:m]==1)[0]
	print idx_d_1, idx_d_2

	idx_w_1 = np.where(labels[m:]==0)[0]
	b = len(idx_w_1)
	idx_w_2 = np.where(labels[m:]==1)[0]
	print idx_w_1, idx_w_2
	# print a, b
	B = np.vstack((A[idx_d_1], A[idx_d_2])).T
	B = np.vstack((B[idx_w_1], B[idx_w_2])).T

	# Check the Laplacian
	# L = np.vstack((np.hstack((np.zeros((m,m)),A)), np.hstack((A.T, np.zeros((n, n))))))
	L = np.vstack((np.hstack((np.diag(np.sum(A, axis=1)),-A)), np.hstack((-A.T, np.diag(np.sum(A, axis=0))))))

	# print e
	return B

