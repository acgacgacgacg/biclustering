import numpy as np 


# Input A: a relational matrix; delta: hyper parameter to control cluster
# Output B: one clustered ralational matrix of A
def delta_bicluster(A, delta):
	I, J = A.shape
	# Initialization
	a_iJ = np.mean(A, axis=0).reshape((1, -1))
	a_Ij = np.mean(A, axis=1).reshape((-1, 1))
	a_IJ = np.mean(A)
	H = np.sum((A-a_iJ-a_Ij+a_IJ)**2) / float(I*J)

	mask_i = np.ones(I, dtype=bool)
	mask_j = np.ones(J, dtype=bool)
	# Iterate until find a delta-square
	while H>delta:
		# try to remove a row or column
		H_i = np.sum((A[mask_i][:, mask_j]-a_iJ-a_Ij+a_IJ)**2, axis=1) / float(a_iJ.shape[1])
		max_i = np.max(H_i)
		H_j = np.sum((A[mask_i][:, mask_j]-a_iJ-a_Ij+a_IJ)**2, axis=0) / float(a_Ij.shape[0])
		max_j = np.max(H_j)
		if max_i>max_j:
			i = np.argmax(H_i)
			c = 0
			for a in range(I):
				if mask_i[a]: c+=1
				if c == i+1: 
					mask_i[a]=False
					break

		else:
			j = np.argmax(H_j)
			c = 0
			for a in range(J):
				if mask_j[a]: c+=1
				if c == j+1: 
					mask_j[a]=False
					break

		# Update a_iJ, a_Ij
		a_iJ = np.mean(A[mask_i][:, mask_j], axis=0).reshape((1, -1))
		a_Ij = np.mean(A[mask_i][:, mask_j], axis=1).reshape((-1, 1))
		a_IJ = np.mean(A[mask_i][:, mask_j])

		# update H
		H_old = H
		H = np.sum((A[mask_i][:, mask_j]-a_iJ-a_Ij+a_IJ)**2) / float(a_iJ.shape[1]*a_Ij.shape[0])

		if H_old-H < 1e-7: break

	print A[mask_i][:, mask_j]

	idx_i_T = np.where(mask_i==True)
	idx_i_F = np.where(mask_i==False)
	idx_j_T = np.where(mask_j==True)
	idx_j_F = np.where(mask_j==False)
	B = np.vstack((A[idx_i_T], A[idx_i_F])).T
	B = np.vstack((B[idx_j_T], B[idx_j_F])).T
	print mask_i
	print mask_j
	return B

