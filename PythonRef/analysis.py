import numpy as np
from scipy.linalg import norm
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import spike as spike
import matplotlib.pylab as plt
from timeit import default_timer as timer

def performance_analysis(A, rhs, filename, pmin=1, pmax=10):
	'''
	Calcula el tiempo de ejecucion en function del numero de particiones
	'''

	partitions = np.arange(pmin, pmax)
	factor_t = np.zeros_like(partitions, dtype=float)
	solve_t  = np.zeros_like(partitions, dtype=float)
	block_dims  = np.zeros_like(partitions)
	m,n        = A.shape
	testiters  = 50


	for i in range(len(partitions)):
		print 'Computing %d-th partition' % (i)
		
		for iter in range(testiters):
			blockdim = np.floor( m / (1. * partitions[i]))

			block = A[0:blockdim, 0:blockdim]

			start_t = timer()
			LU = sla.splu(block)
			end_t = timer()

			print '\tfactor time %.5f' % (end_t - start_t)
			factor_t[i] += end_t - start_t


			start_t = timer()
			x = LU.solve(rhs[0:blockdim, :])
			end_t = timer()

			print '\tsolve time %.5f' % (end_t - start_t)

			solve_t[i] += end_t - start_t

	factor_t /= (1. * testiters)
	solve_t  /= (1. * testiters)

	np.savetxt( filename, np.vstack((partitions.astype(float), factor_t, solve_t)))

	plt.plot(partitions, factor_t, label='factorization')
	plt.plot(partitions, solve_t, label='solve')
	plt.show()

	plt.plot(partitions, np.multiply(factor_t, partitions), label='factorization')
	plt.plot(partitions, np.multiply(solve_t, partitions), label='solve')
	plt.show()