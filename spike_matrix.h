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


matrix_t* matrix_LoadCSR    (const char* filename);
void      matrix_Deallocate (matrix_t* M);
void      matrix_Print      (matrix_t* M, const char* msg);

/* matrix manipulation routines */
matrix_t* matrix_ExtractBlock ( matrix_t* M, 
								const integer_t r0, 
								const integer_t rf,
								const integer_t c0,
								const integer_t cf);

