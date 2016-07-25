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
#include "spike_mpi.h"
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
	MPI_Status  status;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

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

	sm_schedule_t* schedule;

	if(rank == master){ //MASTER
		//matrix_PrintAsDense( A, NULL );
		start_t = GetReferenceTime();
		schedule = spike_solve_analysis( A, nrhs, size-1); //Number of partitions
		matrix_t* R = matrix_CreateEmptyReducedSystem( schedule->p, schedule->n, schedule->ku, schedule->kl);	

		/* -------------------------------------------------------------------- */
		/* .. Factorization Phase. */
		/* -------------------------------------------------------------------- */
		for(integer_t p=0; p<schedule->p; p++)
		{
			integer_t r0,rf,c0,cf;

			r0 = schedule->n[p];
			rf = schedule->n[p+1];

			matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
			
			//MPI
			IsendMatrix(Aij, p+1);
			//End MPI

			//sprintf( msg, "%d-th matrix block", p);
			matrix_PrintAsDense( Aij, "Master");

			matrix_Deallocate(Aij);
		}

		fprintf(stderr, "\nProgram finished\n");
	}
	else{ //slaves
		matrix_t* Aij = recvMatrix(master);
		matrix_PrintAsDense( Aij, "Slave");
	}
	MPI_Finalize();
	return 0;

}
