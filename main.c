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
			fprintf(stderr, "Factorizando primer bloque\n");
			block_t* Vi   = block_Extract(A, r0, rf, rf, rf + A->ku);

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, NULL, Vi->aij, Aij->n, Vi->m );

			block_Deallocate(Vi);
		}
		else if ( p == (schedule->p -1)){
			fprintf(stderr, "Factorizando ultimo bloque\n");
			block_t* Wi = block_Extract(A, r0, rf, r0 - A->kl, r0);

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, NULL, Wi->aij, Aij->n, Wi->m );

			block_Deallocate( Wi );
		}
		else{
			fprintf(stderr, "Factorizando bloque numero %d\n", p);
			block_t* Vi   = block_Extract(A, r0, rf, rf, rf + A->ku);
			block_t* Wi = block_Extract(A, r0, rf, r0 - A->kl, r0);

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, NULL, Vi->aij, Aij->n, Vi->m );
			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, NULL, Wi->aij, Aij->n, Wi->m );

			block_Deallocate( Vi);
			block_Deallocate( Wi);
		}

		matrix_Deallocate(Aij);
	}


	schedule_Destroy(schedule);
	matrix_Deallocate( A );



	fprintf(stderr, "\nProgram finished\n");

	return 0;
}
