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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nShared Memory Spike Solver.");

	/* ================================= */
	integer_t nrhs = 1;
	integer_t p;
	Error_t error;
	char msg[200];


	/* ================================= */
	matrix_t* A = matrix_LoadCSR("Tests/dummy/tridiagonal.bin");

	sm_schedule_t* schedule = spike_solve_analysis( A, nrhs );

	// matrix_t* R = matrix_CreateEmptyReduced(p, n, ku, kl);

	/* ======== FACTORIZATION PHASE ======== */
	for(p=0; p<schedule->p; p++)
	{
		integer_t r0,rf,c0,cf;

		r0 = schedule->interval[p].r0;
		rf = schedule->interval[p].rf;

		matrix_t* Aij = matrix_Extract(A, r0, rf, r0, rf);

		sprintf( msg, "%d-th matrix block", p);
		matrix_Print( Aij, msg);


		if ( p == 0 ){
			fprintf(stderr, "Factorizando primer bloque...\n");
			block_t* Bi    = block_Empty( rf - r0, A->ku, (blocktype_t) _V_BLOCK_ );
			block_t* Vi    = block_Extract(A, r0, rf, rf, rf + A->ku, (blocktype_t) _V_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Bi->aij, Vi->aij, Aij->n, Vi->m );

			block_Deallocate( Vi );
			block_Deallocate( Bi );
		}
		else if ( p == (schedule->p -1)){
			fprintf(stderr, "Factorizando ultimo bloque...\n");
			block_t* Ci = block_Empty( rf - r0, A->kl, (blocktype_t) _W_BLOCK_ );
			block_t* Wi = block_Extract(A, r0, rf, r0 - A->kl, r0, (blocktype_t) _W_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Ci->aij, Wi->aij, Aij->n, Wi->m );

			block_Deallocate( Wi );
			block_Deallocate( Ci );
		}
		else{
			fprintf(stderr, "Factorizando bloque numero %d\n", p);

			fprintf(stderr, "\tFactorizando bloque derecho...\n");
			block_t* Bi    = block_Empty( rf - r0, A->ku, (blocktype_t) _V_BLOCK_ );
			block_t* Vi    = block_Extract(A, r0, rf, rf, rf + A->ku, (blocktype_t) _V_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Bi->aij, Vi->aij, Aij->n, Vi->m );

			fprintf(stderr, "\tFactorizando bloque izquierdo...\n");
			block_t* Ci = block_Empty( rf - r0, A->kl, (blocktype_t) _W_BLOCK_ );
			block_t* Wi = block_Extract(A, r0, rf, r0 - A->kl, r0, (blocktype_t) _W_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Ci->aij, Wi->aij, Aij->n, Wi->m );

			block_Deallocate( Vi );
			block_Deallocate( Wi );
			block_Deallocate( Bi );
			block_Deallocate( Ci );
		}

		matrix_Deallocate(Aij);
	}


	schedule_Destroy(schedule);
	matrix_Deallocate( A );



	fprintf(stderr, "\nProgram finished\n");

	return 0;
}
