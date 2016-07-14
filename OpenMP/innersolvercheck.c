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

static Error_t CheckInnerSolverSolution( matrix_t *A, block_t *x, block_t *rhs )
{
	// local variables
	Error_t error;

	fprintf(stderr, "\nSolving original linear system using reference direct solver");

	error = system_solve( A->colind, A->rowptr, A->aij, x->aij, rhs->aij, A->n, rhs->m );

	block_Print(x, "Solution of the system");

	return (SPIKE_SUCCESS);
};

int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nInner direct solver test.\n");

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	integer_t nrhs;
	Error_t error;


	/* -------------------------------------------------------------------- */
	/* .. Load reference matrix from file. */
	/* -------------------------------------------------------------------- */
	matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/matrix.bin");
	matrix_PrintAsDense(A, "Test matrix");
	
	/* -------------------------------------------------------------------- */
	/* .. Single RHS case */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 1: solve a simple system with a single"
					" vector on the RHS\n");
	nrhs = 1;
	block_t*  x1 = block_Empty( A->n, nrhs, (blocktype_t) _RHS_BLOCK_ );
	block_t*  b1 = block_Empty( A->n, nrhs, (blocktype_t) _RHS_BLOCK_ );

	block_InitializeToValue( x1, __zero ); // solution of the system
	block_InitializeToValue( b1, __unit ); // rhs of the system

	CheckInnerSolverSolution( A, x1, b1);
	
	block_Deallocate ( x1 );
	block_Deallocate ( b1 );

	/* -------------------------------------------------------------------- */
	/* .. Solve the system for RHS with two columns.                        */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 2: solve a simple system with two vectors"
		            " on a single RHS\n");
	nrhs = 2;
	block_t*  x2 = block_Empty( A->n, nrhs, (blocktype_t) _RHS_BLOCK_ );
	block_t*  b2 = block_Empty( A->n, nrhs, (blocktype_t) _RHS_BLOCK_ );

	block_InitializeToValue( x2, __zero ); // solution of the system
	block_InitializeToValue( b2, __unit ); // rhs of the system

	for(integer_t i=0; i < (A->n * nrhs); i++ ){
		if   (i < A->n ) b2->aij[i] =  1.0;
		else             b2->aij[i] = -1.0;
	}

	block_Print( b2, "RHS of the test");

	CheckInnerSolverSolution( A, x2, b2);
	
	block_Deallocate ( x2 );
	block_Deallocate ( b2 );	


	/* -------------------------------------------------------------------- */
	/* .. Extracts a submatrix from the system and solve the system       . */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 3: extracts a submatrix and solves the system"
					" with multiple vectors on a single RHS\n");

	compute_bandwidth(A);
	nrhs = A->ku;
	matrix_t* Aij = matrix_ExtractMatrix(A, 0, 5, 0, 5);
	block_t* x3   = block_Empty( 5, nrhs, (blocktype_t) _V_BLOCK_ );
	block_t* b3   = matrix_ExtractBlock(A, 0, 5, 5, 5 + nrhs, (blocktype_t) _V_BLOCK_ );

	block_InitializeToValue( x3, __zero ); // solution of the system
	block_Print( b3, "RHS extracted from the matrix");

	CheckInnerSolverSolution( Aij, x3, b3);
	
	block_Deallocate ( x3 );
	block_Deallocate ( b3 );
	matrix_Deallocate( Aij);	

	/* -------------------------------------------------------------------- */
	/* .. Clean up and resume                                             . */
	/* -------------------------------------------------------------------- */

	matrix_Deallocate( A );
	fprintf(stderr, "\nTest finished finished\n");

	return 0;
}
