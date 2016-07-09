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
#include <string.h>

/* sparse CSR matrix structure */
typedef struct
{
	integer_t n;
	integer_t nnz;

	integer_t ku;
	integer_t kl;
	integer_t K;

	integer_t* colind; // TODO attribute aligned
	integer_t* rowptr; // TODO attribute aligned
	complex_t* aij;    // TODO attribute aligned

} matrix_t;

/* dense block structure */
typedef enum{ _V_BLOCK_, _W_BLOCK_, _RHS_BLOCK_ } blocktype_t;

typedef struct
{
	blocktype_t  type;
	integer_t    n;
	integer_t    m;
	complex_t   *aij;

} block_t;


matrix_t* matrix_LoadCSR    (const char* filename);
static matrix_t* matrix_CreateEmpty( const integer_t n, const integer_t nnz );
void      matrix_Deallocate (matrix_t* M);
void      matrix_Print      (matrix_t* M, const char* msg);

/* matrix manipulation routines */
matrix_t* matrix_ExtractMatrix (  matrix_t* M,
														const integer_t r0,
														const integer_t rf,
														const integer_t c0,
														const integer_t cf);

/* compares two CSR sparse matrices */
Error_t matrix_AreEqual( matrix_t* A, matrix_t* B );

/* Extracts a dense block from a sparse matrix */
block_t* matrix_ExtractBlock (  matrix_t* M,
													const integer_t r0,
													const integer_t rf,
													const integer_t c0,
													const integer_t cf,
												  blocktype_t type
												);


void block_Deallocate (block_t* B);
void block_Print ( block_t* B, const char* msg);
Error_t matrix_PrintAsDense( matrix_t* A, const char* msg);
Error_t block_AreEqual( block_t* A, block_t* B );
block_t* block_Empty( const integer_t n, const integer_t m, blocktype_t type);

Error_t matrix_FillReduced ( const integer_t TotalPartitions,
														 const integer_t CurrentPartition,
                             integer_t     *n,
                             integer_t     *ku,
                             integer_t     *kl,
                             matrix_t      *R,
                             block_t*       B );

matrix_t* matrix_CreateEmptyReduced( const integer_t p, integer_t *n, integer_t *ku, integer_t *kl );
Error_t GetNnzAndRowsUpToPartition ( const integer_t TotalPartitions, const integer_t CurrentPartition, integer_t *ku, integer_t *kl, integer_t *nnz, integer_t *FirstBlockRow );
static integer_t* ComputeReducedSytemDimensions( integer_t partitions, integer_t *ku, integer_t *kl);
