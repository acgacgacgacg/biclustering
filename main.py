import numpy as np
import matplotlib.pyplot as plt
from delta_bicluster import delta_bicluster
from spectral_bicluster import spectral_bicluster
from info_bicluster import info_bicluster


def main():
	# initialize random adj matrix
	np.random.seed(0)
	A = np.zeros((10, 10))
	A[:5, :5] = 1
	A[5:, 5:] = 1
	# A[7:, :2] = 1
	# Add some noise
	A[2,2] = 0
	A[3, 3] = 0
	A[3, 6:8] = 1
	A[6, 7] =0

	# Shuffle
	A_ = A[np.random.permutation(10)].T[np.random.permutation(10)].T

	np.random.seed()
	# B = delta_bicluster(A_, 0.001)
	# B = spectral_bicluster(A_)
	B = info_bicluster(A_, 2, 2)

	# Display original, shuffled, clusterd matrix
	fig, axes = plt.subplots(1,3)
	axes.flat[0].imshow(A, interpolation='None', cmap='Greys')
	axes.flat[1].imshow(A_, interpolation='None', cmap='Greys')
	axes.flat[2].imshow(B, interpolation='None', cmap='Greys')
	plt.show()


main()
