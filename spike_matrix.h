/*
 * =====================================================================================
 *
 *       Filename:  spike_matrix.h
 *
 *    Description:  Sparse matrix definition for the solver
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:51:07
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */
#include "spike_memory.h"
#include "spike_datatypes.h"
#include <math.h>

/* sparse CSR matrix structure */
typedef struct
{
	integer_t n;
	integer_t nnz;

	integer_t ku;
	integer_t kl;
	integer_t K;

	integer_t* colind;
	integer_t* rowptr;
	complex_t* aij;

} matrix_t;

/* dense block structure */
typedef struct
{
	integer_t  n, m;
	complex_t* aij;

} block_t;


matrix_t* matrix_LoadCSR    (const char* filename);
void      matrix_Deallocate (matrix_t* M);
void      matrix_Print      (matrix_t* M, const char* msg);

/* matrix manipulation routines */
matrix_t* matrix_Extract (  matrix_t* M,
														const integer_t r0,
														const integer_t rf,
														const integer_t c0,
														const integer_t cf);

/* compares two CSR sparse matrices */
Error_t matrix_AreEqual( matrix_t* A, matrix_t* B );

/* Extracts a dense block from a sparse matrix */
block_t* block_Extract (  matrix_t* M,
													const integer_t r0,
													const integer_t rf,
													const integer_t c0,
													const integer_t cf);


void block_Deallocate (block_t* B);
void block_Print ( block_t* B, const char* msg);
Error_t block_AreEqual( block_t* A, block_t* B );
block_t* block_Empty( const integer_t m, const integer_t n);
