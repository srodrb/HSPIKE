/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  SPIKE usage demonstration
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:32:39
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#include "spike_analysis.h"
#include "spike_datatypes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

static Error_t SolveOriginalSystem( matrix_t *A, block_t *x, block_t *rhs )
{
	// local variables
	double start_t, end_t;
	Error_t error;

	fprintf(stderr, "\nSolving original linear system using reference direct solver");

	start_t = GetReferenceTime();
	error = system_solve( A->colind, A->rowptr, A->aij, x->aij, rhs->aij, A->n, rhs->m );
	end_t = GetReferenceTime();

	fprintf(stderr, "\nReference direct solver took %.6lf seconds", end_t - start_t );

	return (SPIKE_SUCCESS);
};

int main(int argc, char *argv[])
{

	MPI_Init (&argc, &argv);	
	int rank, size, master = 0;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	
	//printf("Rank %d of %d\n", rank, size);

	if (rank == master) fprintf(stderr, "\nShared Memory Spike Solver.\n");

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	double start_t, end_t;
	const integer_t nrhs = 1;
	Error_t error;
	char msg[200];


	/* -------------------------------------------------------------------- */
	/* .. Load and initalize the system Ax=f. */
	/* -------------------------------------------------------------------- */
	matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");
	block_t*  x = block_Empty( A->n, nrhs, (blocktype_t) _RHS_BLOCK_ );
	block_t*  f = block_Empty( A->n, nrhs, (blocktype_t) _RHS_BLOCK_ );

	block_InitializeToValue( x, __zero ); // solution of the system
	block_InitializeToValue( f, __unit ); // rhs of the system

#undef _SOLVE_ONLY_WITH_REF_
#ifdef _SOLVE_ONLY_WITH_REF_
	SolveOriginalSystem( A, x, f);
	matrix_Deallocate( A );
	block_Deallocate( x );
	block_Deallocate( f );
	return 0;
#endif
	
	sm_schedule_t* schedule;
	int sendCount;
	if(rank == master){
		matrix_PrintAsDense( A, NULL );
		start_t = GetReferenceTime();
		schedule = spike_solve_analysis( A, nrhs, size);
		matrix_t* R = matrix_CreateEmptyReduced( schedule->p, schedule->n, schedule->ku, schedule->kl);


		/* -------------------------------------------------------------------- */
		/* .. MPI: Calculating number of bytes to send */
		/* -------------------------------------------------------------------- */
		sendCount = (5 + R->nnz + R->n + 1)*4;
		if (_MPI_COMPLEX_T_ == MPI_DOUBLE) sendCount += (R->nnz * _MPI_COUNT_)*8;
		else if( _MPI_COMPLEX_T_ == MPI_FLOAT) sendCount += (R->nnz * _MPI_COUNT_)*4;
		printf("Try: %d \n", sendCount);

	

		/* -------------------------------------------------------------------- */
		/* .. Factorization Phase. */
		/* -------------------------------------------------------------------- */
		for(integer_t p=0; p<schedule->p; p++)
		{
			integer_t r0,rf,c0,cf;

			r0 = schedule->n[p];
			rf = schedule->n[p+1];

			matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);

			//sprintf( msg, "%d-th matrix block", p);
			//matrix_Print( Aij, msg);

			matrix_Deallocate(Aij);
		}

		fprintf(stderr, "\nProgram finished\n");
	}
	MPI_Finalize();
	return 0;
}
