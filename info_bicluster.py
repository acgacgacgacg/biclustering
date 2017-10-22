import numpy as np 

# Input: A: matrix
#		 k: number of row clusters
#		 l: number of column clusters
def info_bicluster(A, k, l):
	m, n = A.shape
	# convert A to joint probability matrix
	SUM = np.sum(A)
	P = A / float(SUM)
	P_X = np.sum(P, axis=1)
	P_Y = np.sum(P, axis=0)
	P_Y_X = P.T / np.sum(P, axis=1)
	P_X_Y = P / np.sum(P, axis=0)
	P_P_SUM = np.sum(np.multiply(P, np.where(P != 0, np.log(P), 0)))

	# Initialize C_X and C_Y
	C_X = np.random.randint(0, k, m)
	C_Y = np.random.randint(0, l, n)

	DL_XY_old = 0

	while True:
		q_X_Xhat = np.zeros((m, k))
		q_Y_Yhat = np.zeros((n, l))
		#-------------------------------------------#
		#		Row partition						#
		#-------------------------------------------#
		C = np.zeros((k, l))
		for i in range(m):
			for j in range(n):
				C[C_X[i], C_Y[j]]+=P[i, j]

		p_Xhat = np.sum(C, axis=1)
		for i, x in zip(range(m), C_X):
			q_X_Xhat[i, x] = P_X[i]/p_Xhat[x]

		p_Yhat = np.sum(C, axis=0)
		for j, y in zip(range(n), C_Y):
			q_Y_Yhat[j, y] = P_Y[j]/p_Yhat[y]

		q_Y_Xhat = np.dot(q_Y_Yhat, np.divide(C.T, np.sum(C, axis=1), out=np.zeros_like(C.T), where=np.sum(C, axis=1)!=0))

		# Check the convergence condition
		Q = np.dot(np.dot(q_X_Xhat, C), q_Y_Yhat.T)

		DL_XY = P_P_SUM - np.sum(np.multiply(P, np.where(Q!=0, np.log(Q), 0)))
		print DL_XY
		if DL_XY_old != 0 and DL_XY_old - DL_XY < 1e-3:
			break
		DL_XY_old = DL_XY

		# Calculate the KL-Divergence for row
		DL_X = np.sum(np.multiply(P_Y_X, np.where(P_Y_X!=0, np.log(P_Y_X), 0)), axis=0).reshape(m, 1) - np.dot(P_Y_X.T, np.where(q_Y_Xhat!=0, np.log(q_Y_Xhat), 0))
		assert DL_X.shape == (m, k), 'Shape mismatch'

		# Reassign for each row
		C_X = np.argmin(np.where(DL_X>0, DL_X, np.inf), axis=1)


		#-------------------------------------------#
		#		Column partition					#
		#-------------------------------------------#
		C = np.zeros((k, l))
		for i in range(m):
			for j in range(n):
				C[C_X[i], C_Y[j]]+=P[i, j]

		p_Xhat = np.sum(C, axis=1)
		for i, x in zip(range(m), C_X):
			q_X_Xhat[i, x] = P_X[i]/p_Xhat[x]

		p_Yhat = np.sum(C, axis=0)
		for j, y in zip(range(n), C_Y):
			q_Y_Yhat[j, y] = P_Y[j]/p_Yhat[y]

		q_X_Yhat = np.dot(q_X_Xhat, np.divide(C, np.sum(C, axis=0), out=np.zeros_like(C), where=np.sum(C, axis=0)!=0))

		# Calculate the KL-Divergence for column
		DL_Y = np.sum(np.multiply(P_X_Y, np.where(P_X_Y!=0, np.log(P_X_Y), 0)), axis=0).reshape(n, 1) - np.dot(P_X_Y.T, np.where(q_X_Yhat!=0, np.log(q_X_Yhat), 0))
		assert DL_Y.shape == (n, l), 'Shape mismatch'

		# Reassign for each row
		C_Y = np.argmin(np.where(DL_Y>0, DL_Y, np.inf), axis=1)


	# Output cluseterd matrix
	idx_X = []
	idx_Y = []
	for i in range(k):
		a = list(np.where(C_X==i)[0])
		idx_X += a
	for j in range(l):
		b = list(np.where(C_Y==j)[0])
		idx_Y += b
	print 'C_X: ', C_X
	print 'C_Y: ', C_Y
	return A[idx_X].T[idx_Y].T

