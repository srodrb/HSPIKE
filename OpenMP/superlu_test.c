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

#include "spike_interfaces.h"
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
	matrix_t 	*Aij;


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
	superlu_solve( A->n, A->nnz, nrhs, A->colind, A->rowptr, A->aij , x1->aij, b1->aij);

	/* show the solution of the system */
	block_Print( x1, "Solution of the linear system");

	/* check the residual */
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

	for(integer_t i=0; i < (A->n * nrhs); i++ ){
		if   (i < A->n ) b2->aij[i] =  1.0;
		else             b2->aij[i] = -1.0;
	}

	/* solve the linear system using the highest level call */
	superlu_solve( A->n, A->nnz, nrhs, A->colind, A->rowptr, A->aij , x2->aij, b2->aij);

	/* show residual */
	block_Print( x2, "Solutiom of the linear system");

	/* check the residual */
	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x2->aij, b2->aij, A->n, b2->m );

	
	/* clean up and resume */
	block_Deallocate ( x2 );
	block_Deallocate ( b2 );	
	

	/* -------------------------------------------------------------------- */
	/* .. CASE 4 Factorizes the matrix and then solves for multiple RHS   . */
	/* ..        separately                                               . */
	/* -------------------------------------------------------------------- */


	/* -------------------------------------------------------------------- */
	/* .. Clean up and resume                                             . */
	/* -------------------------------------------------------------------- */

	matrix_Deallocate( A );
	fprintf(stderr, "\nTest finished finished\n");

	return 0;
}
