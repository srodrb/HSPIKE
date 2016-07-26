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

static block_t* block_Synthetic(const integer_t n,
                                const integer_t m,
                                const integer_t ku,
                                const integer_t kl, 
                                const complex_t value, 
                                blocktype_t type,
                                blocksection_t section )
{

	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));

	B->type = type;
	B->section = section;
	B->n    = n;
	B->m    = m;
	B->ku   = ku;
	B->kl   = kl;
	B->aij  = (complex_t*) spike_malloc( ALIGN_COMPLEX, n * m, sizeof(complex_t));

	for (integer_t row = 0; row < n; row++) {
		for(integer_t col = 0; col < m; col++ ) {
			B->aij[row + col * n] = (0.1 * (row +1)) + value;
		}	
	}

  return (B);
};

static Error_t SolveOriginalSystem( matrix_t *A, block_t *x, block_t *rhs )
{
	// local variables
	double start_t, end_t;
	Error_t error;

	//fprintf(stderr, "\nSolving original linear system using reference direct solver");

	start_t = GetReferenceTime();
	error = system_solve( A->colind, A->rowptr, A->aij, x->aij, rhs->aij, A->n, rhs->m );
	end_t = GetReferenceTime();

	//fprintf(stderr, "\nReference direct solver took %.6lf seconds", end_t - start_t );

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

	double start_t, end_t;
	const integer_t nrhs = 1;
	Error_t error;
	char msg[200];

	//matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");
	matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large.bin");

	integer_t  p     = size - 1;
	integer_t  ku[3] = {2, 1, 1};
	integer_t  kl[3] = {2, 1, 1};
	integer_t  n [4] = {0, 5, 10, 15};


	sm_schedule_t* schedule;
	debug("------------------------------------------------");
	if(rank == master){ //MASTER
		start_t = GetReferenceTime();
		schedule = spike_solve_analysis( A, nrhs, size-1); //Number of partitions
		debug("Values of schedule: max_m:%d, max_n:%d, p:%d ", schedule->max_m, schedule->max_n, schedule->p);
		matrix_t* R = matrix_CreateEmptyReducedSystem( schedule->p, schedule->n, schedule->ku, schedule->kl);
		integer_t r0,rf,c0,cf;
		
		for(integer_t p=0; p<schedule->p; p++){
			int i, check = 0;
			sendSchedulePacked(schedule, p+1);
			sm_schedule_t* sTest = recvSchedulePacked(p+1);
			check += schedule->p - sTest->p;
			check += schedule->max_m - sTest->max_m;
			check += schedule->max_n - sTest->max_n;
			for(i=0; i<schedule->p+1; i++) check += schedule->n[i] - sTest->n[i];
			for(i=0; i<schedule->p+1; i++) check += schedule->r[i] - sTest->r[i];
			for(i=0; i<schedule->p  ; i++) check += schedule->ku[i] - sTest->ku[i];
			for(i=0; i<schedule->p  ; i++) check += schedule->kl[i] - sTest->kl[i];

			if(check == 0) printf("TEST Send Schedule: \t\t%d PASSED\n", p+1);
			else printf("TEST Send Schedule %d : \t\t%d FAIL \n", check, p+1);
						
		}
		for(integer_t p=0; p<schedule->p; p++){

			r0 = schedule->n[p];
			rf = schedule->n[p+1];

			matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
			
			sendMatrix(Aij, p+1);
			matrix_t* test = recvMatrix(p+1);
		
			if(matrix_AreEqual (test, Aij))printf("TEST send Matrix: \t\t%d PASSED\n", p+1);

		}

		for(integer_t p=0; p<schedule->p; p++){

			r0 = schedule->n[p];
			rf = schedule->n[p+1];

			matrix_t* Aij2 = matrix_ExtractMatrix(A, r0, rf, r0, rf);
			
			sendMatrixPacked(Aij2, p+1);
			matrix_t* test2 = recvMatrixPacked(p+1);
		
			if(matrix_AreEqual (test2, Aij2))printf("TEST send Matrix Packed: \t%d PASSED\n", p+1);

		}

		for(integer_t p=0; p<schedule->p; p++){

			r0 = schedule->n[p];
			rf = schedule->n[p+1];

			matrix_t* Aij3 = matrix_ExtractMatrix(A, r0, rf, r0, rf);
			
			IsendMatrix(Aij3, p+1);
			matrix_t* test3 = recvMatrix(p+1);
		
			if(matrix_AreEqual (test3, Aij3))printf("TEST Isend Matrix Packed: \t%d PASSED\n", p+1);

		}

		//Block Test
		for(integer_t p=0; p<schedule->p; p++){
			block_t *V0Test = block_Synthetic( 5, 2, ku[0], kl[0], 2.0, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t *V0 = recvBlock(p+1);
			
			if(block_AreEqual (V0Test, V0))printf("TEST Send Block: \t\t%d PASSED\n", p+1);
		}

		for(integer_t p=0; p<schedule->p; p++){
			block_t *V0Test = block_Synthetic( 5, 2, ku[0], kl[0], 2.0, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t *V0 = recvBlockPacked(p+1);
			if(block_AreEqual (V0Test, V0))printf("TEST Send Block Packed: \t%d PASSED\n", p+1);
		}
	}
	else{
		sm_schedule_t* sTest = recvSchedulePacked(master);
		sendSchedulePacked(sTest, master);
		
		//Matrix Testing
		matrix_t* Aij = recvMatrix(master);
		sendMatrix(Aij, master);
		Aij = recvMatrixPacked(master);
		sendMatrixPacked(Aij, master);
		Aij = recvMatrix(master);
		IsendMatrix(Aij, master);

		//Block Testing
		block_t *V0 = block_Synthetic( 5, 2, ku[0], kl[0], 2.0, _V_BLOCK_, _WHOLE_SECTION_ );
		sendBlock(V0, master);
		sendBlockPacked(V0, master);

	}
	MPI_Finalize();
	return 0;
}
