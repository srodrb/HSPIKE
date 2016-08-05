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

// #include "spike_analysis.h"
#include "spike_interfaces.h"
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
	error   = directSolver_Solve  ( A->n,
									A->nnz,
									rhs->m,
									A->colind, 
									A->rowptr, 
									A->aij,
									x->aij,
									rhs->aij );
	end_t   = GetReferenceTime();

	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, rhs->aij, A->n, rhs->m );

	fprintf(stderr, "\nReference direct solver took %.6lf seconds", end_t - start_t );

	return (SPIKE_SUCCESS);
};


int main(int argc, const char *argv[])
{
	CheckPreprocessorMacros();

	fprintf(stderr, "\nShared Memory Spike Solver.\n");

	/* -------------------------------------------------------------------- */
	/* .. Load and initalize the system Ax=f. */
	/* -------------------------------------------------------------------- */
	const integer_t nrhs = 1;
	// matrix_t* A = matrix_LoadCSR("../Tests/spike/penta_10e7.d");
	// matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large_10e6.d");
	// matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large.bin");
	// matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/small.bin");
	// matrix_t* A = matrix_LoadCSR("../Tests/spike/penta_15.bin");
	matrix_t* A = matrix_LoadCSR("../Tests/spike/penta_15.z");
	// matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");

	block_t*  x = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  f = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x, __zero  ); // solution of the system
	block_InitializeToValue( f, __punit ); // rhs of the system
	
	/* -------------------------------------------------------------------- */
	/* .. Call the direct solver using the high-level interface           . */
	/* -------------------------------------------------------------------- */
	zspike_core_host(A->n, A->nnz, nrhs, A->colind, A->rowptr, (complex16 *restrict) A->aij, (complex16 *restrict) x->aij, (complex16 *restrict) f->aij);


	/* -------------------------------------------------------------------- */
	/* .. Check residual and compare against reference solver             . */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nPARDISO REFERENCE SOLUTION...\n");
	block_InitializeToValue( x, __zero  ); // solution of the system
	block_InitializeToValue( f, __punit ); // rhs of the system
	SolveOriginalSystem( A, x, f);
	block_Print(x, "Solucion del sistema");

	/* -------------------------------------------------------------------- */
	/* .. Clean up. */
	/* -------------------------------------------------------------------- */
	matrix_Deallocate ( A );
	block_Deallocate  ( x );
	block_Deallocate  ( f );
	

	/* -------------------------------------------------------------------- */
	/* .. Load and initalize the system Ax=f. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "Number of malloc() calls %d, number of free() calls %d\n", cnt_alloc, cnt_free );

	fprintf(stderr, "\nProgram finished\n");
	return 0;
}
