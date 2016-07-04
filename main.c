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

int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nShared Memory Spike Solver.");

	/* ================================= */
	integer_t nrhs = 1;
	integer_t p;


	/* ================================= */
	matrix_t* A = matrix_LoadCSR("test_0.bin");

	sm_schedule_t* schedule = spike_solve_analysis( A, nrhs );	

	/* ======== FACTORIZATION PHASE ======== */
	for(p=0; p<schedule->p; p++)
	{
		integer_t r0,rf,c0,cf;

		r0 = schedule->interval[p].r0;
		rf = schedule->interval[p].rf;

		// TODO: es interesante calcular el BW localmente
		matrix_t* Aij = matrix_ExtractBlock(A, r0, rf, r0, rf);

		matrix_Deallocate(Aij);
	}


	schedule_Destroy(schedule);
	matrix_Deallocate( A );

	

	fprintf(stderr, "\nProgram finished\n");
	
	return 0;
}
