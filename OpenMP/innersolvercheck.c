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
	matrix_t *Aij;


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
	nrhs = 1;
	block_t*  x1 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b1 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x1, __zero  ); // solution of the system
	block_InitializeToValue( b1, __punit ); // rhs of the system

	CheckInnerSolverSolution( A, x1, b1);
	
	block_Deallocate ( x1 );
	block_Deallocate ( b1 );

	/* -------------------------------------------------------------------- */
	/* .. CASE 2 Solve the system for RHS with two columns.                        */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 2: solve a simple system with two vectors"
		            " on a single RHS\n");

	nrhs = 2;
	block_t*  x2 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b2 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x2, __zero  ); // solution of the system
	block_InitializeToValue( b2, __punit ); // rhs of the system

	for(integer_t i=0; i < (A->n * nrhs); i++ ){
		if   (i < A->n ) b2->aij[i] =  1.0;
		else             b2->aij[i] = -1.0;
	}

	block_Print( b2, "RHS of the test");

	system_solve( A->colind, A->rowptr, A->aij, x2->aij, b2->aij, A->n, b2->m );
	
	block_Deallocate ( x2 );
	block_Deallocate ( b2 );	


	/* -------------------------------------------------------------------- */
	/* .. CASE 3 Extracts a submatrix from the system and solve the system. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 3: extracts a submatrix and solves the system"
					" with multiple vectors on a single RHS\n");

	nrhs = A->ku;
	Aij = matrix_ExtractMatrix(A, 0, 5, 0, 5);
	block_t* x3   = block_CreateEmptyBlock( 5, nrhs, 0, 0,_V_BLOCK_, _WHOLE_SECTION_ );
	block_t* b3   = matrix_ExtractBlock(A, 0, 5, 5, 5 + nrhs, _V_BLOCK_ );

	block_InitializeToValue( x3, __zero ); // solution of the system
	block_Print( b3, "RHS extracted from the matrix");

	CheckInnerSolverSolution( Aij, x3, b3);
	
	block_Deallocate ( x3 );
	block_Deallocate ( b3 );
	matrix_Deallocate( Aij );	

	/* -------------------------------------------------------------------- */
	/* .. CASE 4 Factorizes the matrix and then solves for multiple RHS   . */
	/* ..        separately                                               . */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 4: extracts a submatrix and solves the system"
					" with multiple vectors separately, keeping the LU factors"
					" in memory between triangular sweeps.\n");


	Aij               = matrix_ExtractMatrix   (A, 0, 5, 0, 5);
	block_t* x4       = block_CreateEmptyBlock ( 5, 1, 0, 0,_V_BLOCK_, _WHOLE_SECTION_ );
	block_t* b4_left  = matrix_ExtractBlock    (A, 0, 5, 5, 6, _V_BLOCK_ );
	block_t* b4_right = matrix_ExtractBlock    (A, 0, 5, 6, 7, _V_BLOCK_ );

	matrix_PrintAsDense( Aij, "Coefficient matrix");

	/* pardiso control paramters */
	MKL_INT pardiso_conf[64];

	block_Print( b4_left, "Left-most RHS");
	directSolver_Factorize( Aij->colind, Aij->rowptr, Aij->aij, Aij->n, nrhs, &pardiso_conf);

	directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, x4->aij, b4_left->aij, Aij->n, b4_left->m, &pardiso_conf);
	block_Print( x4, "Solution for the left-most RHS vector");

	directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, x4->aij, b4_right->aij, Aij->n, b4_right->m, &pardiso_conf);
	block_Print( x4, "Solution for the right-most RHS vector");

	directSolver_CleanUp( Aij->colind, Aij->rowptr, Aij->aij, x4->aij, b4_right->aij, Aij->n, b4_right->m, &pardiso_conf);
	matrix_PrintAsDense( Aij, "Coefficient matrix after releasing");


	block_Deallocate( x4 );
	block_Deallocate( b4_left);
	block_Deallocate( b4_right);
	matrix_Deallocate( Aij);

	/* -------------------------------------------------------------------- */
	/* .. Clean up and resume                                             . */
	/* -------------------------------------------------------------------- */

	matrix_Deallocate( A );
	fprintf(stderr, "\nTest finished finished\n");

	return 0;
}
