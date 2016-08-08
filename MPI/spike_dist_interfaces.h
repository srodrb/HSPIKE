#ifndef _SPIKE_MPI_INTERFACES_H_
	#define _SPIKE_MPI_INTERFACES_H_

	
	#include "spike_analysis.h"
	#include "spike_mpi.h"

	/* API functions */
	Error_t spike_dist_blocking( matrix_t *A, block_t *x, block_t *f, const integer_t nrhs);
	Error_t spike_dist_nonblocking( matrix_t *A, block_t *x, block_t *f, const integer_t nrhs );

#endif /* end of  _SPIKE_MPI_INTERFACES_H_ definition */
