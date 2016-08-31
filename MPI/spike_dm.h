#ifndef _SPIKE_MPI_INTERFACES_H_
	#define _SPIKE_MPI_INTERFACES_H_

	
	#include "spike_mpi.h"

	/* API functions */
	Error_t spike_dm( matrix_t *A, block_t *x, block_t *f, const integer_t nrhs );

#endif /* end of  _SPIKE_MPI_INTERFACES_H_ definition */
