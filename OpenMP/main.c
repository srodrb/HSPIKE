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
	matrix_t* A = matrix_LoadCSR("../Tests/spike/penta_15.bin");
	// matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large.bin");
	matrix_PrintAsDense( A, "Original coeffient matrix" );

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

	start_t = GetReferenceTime();

	/* compute an optimal solving strategy */
	sm_schedule_t* S = spike_solve_analysis( A, nrhs, 3 );

	/* create the reduced sytem in advanced, based on the solving strategy */
	matrix_t* R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
	block_t*  xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );


	/* -------------------------------------------------------------------- */
	/* .. Factorization Phase. */
	/* -------------------------------------------------------------------- */
	for(integer_t p=0; p < S->p; p++)
	{
		fprintf(stderr, "Solving %d-th block\n", p);
		
		const integer_t r0 = S->n[p];
		const integer_t rf = S->n[p+1];

		/* allocate pardiso configuration parameters */
		// void *pardiso_conf = (void*) spike_malloc( ALIGN_INT, 64, sizeof(integer_t));
		MKL_INT pardiso_conf[64];

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
		directSolver_Factorize( Aij->colind, Aij->rowptr, Aij->aij, Aij->n, nrhs, &pardiso_conf);

		/* -------------------------------------------------------------------- */
		/* .. Solve Ai * yi = fi                                                */
		/* Extracts the fi portion from f, creates a yi block used as container */
		/* for the solution of the system. Then solves the system.              */
		/* -------------------------------------------------------------------- */
		block_t*  fi  = block_ExtractBlock    ( f, r0, rf );
		block_t*  yi  = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

		block_SetBandwidthValues( fi, A->ku, A->kl );
		block_SetBandwidthValues( yi, A->ku, A->kl );

		/* solve the system for the RHS value */
		directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, yi->aij, fi->aij, Aij->n, nrhs, &pardiso_conf );

		/* Extract the tips of the yi block */
		block_t* yit = block_ExtractTip( yi, _TOP_SECTION_ );
		block_t* yib = block_ExtractTip( yi, _BOTTOM_SECTION_ );

		/* Add the tips of the yi block to the reduced RHS */
		block_AddTipTOReducedRHS( p, S->ku, S->kl, xr, yit );
		block_AddTipTOReducedRHS( p, S->ku, S->kl, xr, yib );

		/* clean up */
		block_Deallocate (fi );
		block_Deallocate (yi );
		block_Deallocate (yit);
		block_Deallocate (yib);

		if ( p == 0 ){
			block_t* Vi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );

			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );

			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);
		}
		else if ( p == ( S->p -1)){
			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m, &pardiso_conf );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );

			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );
			
			sprintf( msg, "%d-th partition, Wi block", p);
			block_Print( Wi, msg );

			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);
		}
		else{
			block_t* Vi    = block_CreateEmptyBlock( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi    = matrix_ExtractBlock   (A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );

			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );

			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);

			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m, &pardiso_conf );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );

			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );
			
			sprintf( msg, "%d-th partition, Wi block", p);
			block_Print( Wi, msg );

			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);
		}

		directSolver_CleanUp(NULL,NULL,NULL,NULL,NULL, Aij->n, nrhs, &pardiso_conf);

		matrix_Deallocate(Aij); fprintf(stderr, "Fin del programa");
	}

	/* -------------------------------------------------------------------- */
	/* .. Solution of the reduced system. */
	/* -------------------------------------------------------------------- */

	block_t* yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );  
	reduced_PrintAsDense( R, yr, xr, "Reduced system");
	system_solve ( R->colind, R->rowptr, R->aij, yr->aij, xr->aij, R->n, xr->m);

	block_Print( yr, "Solution of the reduced system");

	/* Free some memory, yr and R are not needed anymore */
	block_Deallocate ( xr );
	matrix_Deallocate( R  );

	/* -------------------------------------------------------------------- */
	/* .. Backward substitution phase. */
	/* -------------------------------------------------------------------- */
	for(integer_t p=0; p < S->p; p++)
	{
		fprintf(stderr, "Processing backward solution for the %d-th block\n", p);

		/* compute the limits of the blocks */
		const integer_t obs = S->n[p];        	/* original system starting row */
		const integer_t obe = S->n[p+1];	  	/* original system ending row   */
		const integer_t rbs = S->r[p];		  	/* reduceed system starting row */
		const integer_t rbe = S->r[p+1];		/* reduced system ending row    */

		/* allocate pardiso configuration parameters */
		MKL_INT pardiso_conf[64];

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, obs, obe, obs, obe);
		directSolver_Factorize( Aij->colind, Aij->rowptr, Aij->aij, Aij->n, nrhs, &pardiso_conf);

		/* extract fi sub-block */
		block_t*  fi  = block_ExtractBlock(f, obs, obe );
		block_Print ( fi, "Dense fi sub-block");
			
		/* Compute the solution depending on the partition */
		if ( p == 0 ){

			block_t* Bi  = matrix_ExtractBlock ( A, obe - S->ku[p], obe, obe, obe + S->ku[p], _WHOLE_SECTION_ );
			block_Print( Bi, "Dense Bi sub-block");
			
			block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
			block_Print( xt_next, "xt(i+1) dende sub-block");

			// block_t* xit = block_CreateEmptyBlock  ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );



			// directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_Deallocate ( Bi );
			block_Deallocate ( xt_next); 
		}
		else if ( p == ( S->p -1)){

			// block_t* Ci  = matrix_ExtractBlock ( A, obe - S->ku[p], obe, obe, obe + S->ku[p], _WHOLE_SECTION_ );			
			// block_t* xb_prev = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
			
			// block_t* xit = block_CreateEmptyBlock  ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );

			// directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

//			block_Deallocate ( Ci );
//			block_Deallocate ( xb_prev);		
		}
		else{

//			block_t* Bi  = matrix_ExtractBlock ( A, obe - S->ku[p], obe, obe, obe + S->ku[p], _WHOLE_SECTION_ );			
//			block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
//			
//			// block_t* xit = block_CreateEmptyBlock  ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
//
//			// directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );
//
//
//
//			block_t* Ci  = matrix_ExtractBlock ( A, obe - S->ku[p], obe, obe, obe + S->ku[p], _WHOLE_SECTION_ );
//			block_t* xb_prev = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
//			
//			// block_t* xit = block_CreateEmptyBlock  ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
//
//			// directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );
//
//			block_Deallocate ( Bi );
//			block_Deallocate ( Ci );
//			block_Deallocate ( xt_next);			 	
//			block_Deallocate ( xb_prev);
		}

		directSolver_CleanUp(NULL,NULL,NULL,NULL,NULL, Aij->n, nrhs, &pardiso_conf);
		block_Deallocate ( fi );
		matrix_Deallocate(Aij);
		
	}

	/* -------------------------------------------------------------------- */
	/* .. Clean up. */
	/* -------------------------------------------------------------------- */
	schedule_Destroy  ( S );
	matrix_Deallocate ( A );
	block_Deallocate  ( yr);
	block_Deallocate  ( x );
	block_Deallocate  ( f );
	
	// directSolver_CleanUp( pardiso_conf );

	end_t = GetReferenceTime();

	fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);

	/* -------------------------------------------------------------------- */
	/* .. Load and initalize the system Ax=f. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nProgram finished\n");

	return 0;
}
