#ifndef _SPIKE_MPI_INTERFACES_H_
	#define _SPIKE_MPI_INTERFACES_H_

	
	#include "spike_mpi.h"
	#include "spike_blocking.h"
	//#include "spike_mpi.h"

	/* API functions */
	extern block_t* Bib;
	extern block_t* Cit;
	Error_t spike_dm( matrix_t *A, block_t *x, block_t *f, const integer_t nrhs );

#endif /* end of  _SPIKE_MPI_INTERFACES_H_ definition */
