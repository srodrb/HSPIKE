/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Parallel spike with MPI
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:32:39
 *       Revision:  none
 *       Compiler:  mpiicc
 *
 *         Author:  Albert Coca Abelló
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */
/** 
 *	@file main.c 
 */

/*! \mainpage Spike summary
 *
 * \section intro_sec Introduction
 *
 * This is an implementation of spike in distributed memory system.
 * For more information go to spike_mpi.c documentation.
 *
 * \section org_sec Organization
 * Barcelona Supercomputing Center
 */

#include "spike_dm.h"

const integer_t nrhs = 1;

static Error_t SolveOriginalSystem( matrix_t *A, block_t *x, block_t *rhs )
{
	// local variables
	double start_t, end_t;
	Error_t error;

	fprintf(stderr, "\nSolving original linear system using reference direct solver");

	start_t = GetReferenceTime();
	error = directSolver_Solve( A->n, A->nnz, rhs->m, A->colind, A->rowptr, A->aij, x->aij, rhs->aij );
	end_t = GetReferenceTime();

	fprintf(stderr, "\nReference direct solver took %.6lf seconds", end_t - start_t );

	return (SPIKE_SUCCESS);
};

int main(int argc, char *argv[])
{
	/* Set up MPI environment */
	MPI_Init( &argc, &argv );
	int rank, size, master = 0;
	MPI_Status  status;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	
	if ( rank == 0) {
			// matrix_t* A = matrix_LoadCSR("ReducedSystem.bsit");
			// matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");
			// matrix_t* A = matrix_LoadCSR("../Tests/heptadiagonal/medium.bin");
			matrix_t* A = matrix_LoadCSR("../Tests/spike/moreReal.d");
			// matrix_t* A = matrix_LoadCSR("../Tests/complex16/penta_1k.z");
			// matrix_t* A = matrix_LoadCSR("../Tests/spike/15e10Matrix.bin");
			// matrix_t* A = matrix_LoadCSR("../Tests/spike/permuted.bsit");
			// matrix_PrintAsDense( A, "Original coeffient matrix" );

			matrix_PrintAsDense(A, NULL);

			// Compute matrix bandwidth
			block_t*  x = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
			block_t*  f = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

			block_InitializeToValue( x, __zero  ); // solution of the system
			block_InitializeToValue( f, __punit ); // rhs of the system

			/* compute an optimal solving strategy */
			dm_schedule_t* S = spike_solve_analysis( A, nrhs );
			
			/* call MPI solver */
			spike_dm ( A, x, f, nrhs );

			/* Compute residual of the linear system */
			ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
			
			/* compare with MKL's Pardiso only if the system is small.. */
			//if ( A->n < 5000 ) SolveOriginalSystem( A, x, f);

			/* resume and exit */
			matrix_Deallocate( A );
			block_Deallocate ( f );
			block_Deallocate ( x );
			schedule_Destroy( S );
	}
	else {
		/* call MPI solver */
		spike_dm( NULL, NULL, NULL, nrhs );
	}

	MPI_Finalize();
	
	return 0;
}
