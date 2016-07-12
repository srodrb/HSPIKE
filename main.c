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

static Error_t SolveOriginalSystem( matrix_t *A, const integer_t nrhs )
{
	// local variables
	double start_t, end_t;
	Error_t error;

	// create rhs (b) and x vectors
	complex_t *rhs = (complex_t*) spike_malloc( ALIGN_COMPLEX, A->n * nrhs, sizeof(complex_t));
	complex_t *x   = (complex_t*) spike_malloc( ALIGN_COMPLEX, A->n * nrhs, sizeof(complex_t));

	// initialize b to one and x to zero
	for(integer_t i=0; i<(A->n * nrhs); i++ ) {rhs[i] = (complex_t) __unit; }
	memset((void*) x, 0, A->n * nrhs * sizeof(complex_t));

	fprintf(stderr, "\nSolving original linear system using reference direct solver");

	start_t = GetReferenceTime();
	error = system_solve( A->colind, A->rowptr, A->aij, x, rhs, A->n, nrhs );
	end_t = GetReferenceTime();

	fprintf(stderr, "\nReference direct solver took %.6lf seconds", end_t - start_t );

	spike_nullify (rhs);
	spike_nullify (x);

	return (SPIKE_SUCCESS);
};

int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nShared Memory Spike Solver.\n");

	/* ================================= */
	double start_t, end_t;
	integer_t nrhs = 1;
	integer_t p;
	Error_t error;
	char msg[200];


	/* ================================= */
	matrix_t* A = matrix_LoadCSR("Tests/dummy/tridiagonal.bin");
	// SolveOriginalSystem( A, nrhs);
	// matrix_Deallocate( A );
	// return 0;
	// matrix_PrintAsDense( A, NULL );


	start_t = GetReferenceTime();
	sm_schedule_t* schedule = spike_solve_analysis( A, nrhs, 4 );

	matrix_t* R = matrix_CreateEmptyReduced( schedule->p, schedule->n, schedule->ku, schedule->kl);

	/* ======== FACTORIZATION PHASE ======== */
	for(p=0; p<schedule->p; p++)
	{
		integer_t r0,rf,c0,cf;

		r0 = schedule->n[p];
		rf = schedule->n[p+1];

		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);

		// sprintf( msg, "%d-th matrix block", p);
		// matrix_Print( Aij, msg);

		if ( p == 0 ){
			block_t* Vi    = block_Empty( rf - r0, A->ku, (blocktype_t) _V_BLOCK_ );
			block_t* Bi    = matrix_ExtractBlock(A, r0, rf, rf, rf + A->ku, (blocktype_t) _V_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m );

			//sprintf( msg, "%d-th partition, Vi block", p);
			//block_Print( Vi, msg );

			error = matrix_FillReduced( schedule->p, p, schedule->n, schedule->ku, schedule->kl, R, Vi );

			block_Deallocate( Vi );
			block_Deallocate( Bi );
		}
		else if ( p == (schedule->p -1)){
			block_t* Wi = block_Empty( rf - r0, A->kl, (blocktype_t) _W_BLOCK_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, (blocktype_t) _W_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m );
			error = matrix_FillReduced( schedule->p, p, schedule->n, schedule->ku, schedule->kl, R, Wi );

			//sprintf( msg, "%d-th partition, Wi block", p);
			//block_Print( Wi, msg );

			block_Deallocate( Wi );
			block_Deallocate( Ci );
		}
		else{
			block_t* Vi    = block_Empty( rf - r0, A->ku, (blocktype_t) _V_BLOCK_ );
			block_t* Bi    = matrix_ExtractBlock(A, r0, rf, rf, rf + A->ku, (blocktype_t) _V_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m );
			error = matrix_FillReduced( schedule->p, p, schedule->n, schedule->ku, schedule->kl, R, Vi );
;
			block_t* Wi = block_Empty( rf - r0, A->kl, (blocktype_t) _W_BLOCK_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, (blocktype_t) _W_BLOCK_ );

			error = system_solve( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m );
			error = matrix_FillReduced( schedule->p, p, schedule->n, schedule->ku, schedule->kl, R, Wi );

			//sprintf( msg, "%d-th partition, Wi block", p);
			//block_Print( Wi, msg );

			//sprintf( msg, "%d-th partition, Vi block", p);
			//block_Print( Vi, msg );

			block_Deallocate( Vi );
			block_Deallocate( Wi );
			block_Deallocate( Bi );
			block_Deallocate( Ci );
		}

		matrix_Deallocate(Aij);
	}

	// ahora resolvemos el sistema reducido
	// solve_system();

	// recuperamos la solucion del sistema original

	matrix_PrintAsDense( R, "Reduced system");

	schedule_Destroy(schedule);
	matrix_Deallocate( A );

	end_t = GetReferenceTime();

	fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);

	fprintf(stderr, "\nProgram finished\n");

	return 0;
}
