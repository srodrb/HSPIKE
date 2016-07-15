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

int main(int argc, const char *argv[])
{
	CheckPreprocessorMacros();

	fprintf(stderr, "\nShared Memory Spike Solver.\n");

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

	// Compute matrix bandwidth

	block_t*  x = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  f = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

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
	matrix_PrintAsDense( A, NULL );

	start_t = GetReferenceTime();
	sm_schedule_t* S = spike_solve_analysis( A, nrhs, 2 );

	matrix_t* R = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);

	/* -------------------------------------------------------------------- */
	/* .. Factorization Phase. */
	/* -------------------------------------------------------------------- */
	for(integer_t p=0; p < S->p; p++)
	{
		integer_t r0,rf,c0,cf;

		r0 = S->n[p];
		rf = S->n[p+1];

		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);

		// sprintf( msg, "%d-th matrix block", p);
		// matrix_Print( Aij, msg);

		if ( p == 0 ){
			block_t* Vi    = block_CreateEmptyBlock( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi    = matrix_ExtractBlock   ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			system_solve( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m );

			block_t* Vit = block_ExtractBlock( Vi, _TOP_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Vit );

			block_t* Vib = block_ExtractBlock( Vi, _BOTTOM_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Vib );


			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);
		}
		else if ( p == ( S->p -1)){
			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			system_solve( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m );

			block_t* Wit = block_ExtractBlock( Wi, _TOP_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Wit );

			block_t* Wib = block_ExtractBlock( Wi, _BOTTOM_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Wib );
			
			sprintf( msg, "%d-th partition, Wi block", p);
			block_Print( Wi, msg );

			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);
		}
		else{
			block_t* Vi    = block_CreateEmptyBlock( rf - r0, A->ku, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi    = matrix_ExtractBlock   (A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			system_solve( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m );

			block_t* Vit = block_ExtractBlock( Vi, _TOP_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Vit );

			block_t* Vib = block_ExtractBlock( Vi, _BOTTOM_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Vib );


			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);

			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			system_solve( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m );

			block_t* Wit = block_ExtractBlock( Wi, _TOP_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Wit );

			block_t* Wib = block_ExtractBlock( Wi, _BOTTOM_SECTION_ );
			matrix_AddBlockToReducedSystem( S->p, p, S->ku, S->kl, R, Wib );
			
			sprintf( msg, "%d-th partition, Wi block", p);
			block_Print( Wi, msg );

			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);
		}

		matrix_Deallocate(Aij);
	}

	/* -------------------------------------------------------------------- */
	/* .. Solution of the reduced system. */
	/* -------------------------------------------------------------------- */

	// ahora resolvemos el sistema reducido
	// solve_system();

	/* -------------------------------------------------------------------- */
	/* .. Backward substitution phase. */
	/* -------------------------------------------------------------------- */

	matrix_PrintAsDense( R, "Reduced system");

	/* -------------------------------------------------------------------- */
	/* .. Clean up. */
	/* -------------------------------------------------------------------- */
	schedule_Destroy  ( S );
	matrix_Deallocate ( A );
	block_Deallocate  ( x );
	block_Deallocate  ( f );

	end_t = GetReferenceTime();

	fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);

	/* -------------------------------------------------------------------- */
	/* .. Load and initalize the system Ax=f. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nProgram finished\n");

	return 0;
}
