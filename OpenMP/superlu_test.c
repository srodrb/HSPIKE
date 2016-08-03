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
	fprintf(stderr, "\nInner direct solver test.\n");

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	integer_t 	nrhs;
	Error_t 	error;


	/* -------------------------------------------------------------------- */
	/* .. Load reference matrix from file. */
	/* -------------------------------------------------------------------- */
	matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/small.bin");
	matrix_PrintAsDense(A, "Test matrix");
	
	/* -------------------------------------------------------------------- */
	/* .. CASE 1 Single RHS case */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 1: solve a simple system with a single"
					" vector on the RHS\n");

	nrhs         = 1;
	block_t*  x1 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b1 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x1, __zero  ); 
	block_InitializeToValue( b1, __punit ); 

	/* solve the linear system using the highest level call */
	directSolver_Solve( A->n,
						A->nnz,
						nrhs,
						A->colind,
						A->rowptr,
						A->aij,
						x1->aij,
						b1->aij);

	/* show the solution of the system */
	block_Print( x1, "Solution of the linear system");

	/* check residual */
	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x1->aij, b1->aij, A->n, b1->m );
	
	block_Deallocate ( x1 );
	block_Deallocate ( b1 );

	/* -------------------------------------------------------------------- */
	/* .. CASE 2 Solve the system for multiple RHSs.                        */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 2: solve a simple system with two vectors"
		            " on a single RHS\n");

	nrhs         = 2;
	block_t*  x2 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b2 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x2, __zero  ); 
	block_InitializeToValue( b2, __punit );

	for(integer_t i=0; i < (A->n * nrhs); i++ )
		b2->aij[i] = (i < A->n ) ? __punit : __nunit;

	/* solve the linear system using the highest level call */
	directSolver_Solve( A->n,
						A->nnz,
						nrhs,
						A->colind,
						A->rowptr,
						A->aij ,
						x2->aij,
						b2->aij);

	/* show residual */
	block_Print( x2, "Solutiom of the linear system");

	/* check residual */
	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x2->aij, b2->aij, A->n, b2->m );

	
	/* clean up and resume */
	block_Deallocate ( x2 );
	block_Deallocate ( b2 );	
	

	/* -------------------------------------------------------------------- */
	/* .. CASE 3 Factorizes the matrix and then solves for multiple RHS   . */
	/* ..        separately                                               . */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 3: solve a simple system with two vectors"
	            " separately, keeping the factorization in memory.\n");
	
	nrhs         = 2;
	block_t*  x3 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b3 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x3, __zero  ); 
	block_InitializeToValue( b3, __punit );

	for(integer_t i=0; i < (A->n * nrhs); i++ ) b3->aij[i] = (i < A->n ) ? __punit : __nunit;

	matrix_PrintAsDense(A, "Original matrix");

	DirectSolverHander_t *handler = directSolver_CreateHandler();

	directSolver_Configure( handler );

	/* factorize, keeping the LU in memory */
	directSolver_Factorize( handler, A->n, A->nnz, A->colind, A->rowptr, A->aij );

	/* solve for the different packs of RHS */
	directSolver_SolveForRHS( handler, 2, x3->aij, b3->aij);

	// directSolver_SolveForRHS( handler, 1, &x3->aij[A->n], &b3->aij[A->n]);

	/* check residual */
	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x3->aij, b3->aij, A->n, b3->m );

	block_Print(x3, "Solution for test 3");


	directSolver_Finalize ( handler );

	block_Deallocate( x3 );
	block_Deallocate( b3 );


	/* -------------------------------------------------------------------- */
	/* .. Clean up and resume                                             . */
	/* -------------------------------------------------------------------- */

/*	
	block_t* blockref = matrix_ExtractBlock ( A, 0, 5, 5, 7, _RHS_BLOCK_ );
	block_Print( blockref, "reference block");


	matrix_t* Bi = matrix_ExtractMatrix(A, 3, 5, 5, 7);

	matrix_PrintAsDense( Bi, "Bi matrix");

	block_t*  foo = block_BuildBlockFromMatrix( Bi, _V_BLOCK_, 5, 2, 2, 2 );
	block_Print( foo, "suerte...");

	// matrix_Deallocate( Bi );
	block_Deallocate( foo );
	block_Deallocate( blockref);
*/

	matrix_Deallocate( A );
	fprintf(stderr, "\nTest finished finished\n");

		fprintf(stderr, "Number of malloc() %d, number of free() %d\n", cnt_alloc, cnt_free );


	return 0;
}
