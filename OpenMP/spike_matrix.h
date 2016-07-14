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
#include "spike_common.h"
#include "spike_datatypes.h"
#include "mkl.h"
#include <string.h>


 #define _MAX_PRINT_DIMENSION_ 25

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
typedef enum{ _TOP_SECTION_, _CENTRAL_SECTION_, _BOTTOM_SECTION_, _WHOLE_SECTION_ } blocksection_t;

typedef struct
{
	blocktype_t     type;
	blocksection_t  section;
	integer_t       n;
	integer_t       m;
	integer_t       ku;
	integer_t       kl;
	complex_t       *aij;

} block_t;

/* -------------------------------------------------------------------- */
/* .. Functions affecting sparse CSR matrix structure                   */
/* -------------------------------------------------------------------- */

matrix_t*         matrix_LoadCSR                (const char* filename);

matrix_t*         matrix_CreateEmptyMatrix      (const integer_t n, const integer_t nnz );


Error_t           matrix_Deallocate             (matrix_t* M);

Bool_t            matrix_AreEqual               (matrix_t* A, matrix_t* B );

Error_t           matrix_PrintAsSparse          (matrix_t* M, const char* msg);

Error_t           matrix_PrintAsDense           (matrix_t* A, const char* msg);

block_t*          matrix_ExtractBlock 		   (matrix_t* M,
								            	const integer_t r0,
								            	const integer_t rf,
								            	const integer_t c0,
								            	const integer_t cf,
								            	blocktype_t type);

matrix_t*         matrix_ExtractMatrix 		   (matrix_t* M,
												const integer_t r0,
												const integer_t rf,
												const integer_t c0,
												const integer_t cf );

/* -------------------------------------------------------------------- */
/* .. Functions affecting block structures.                             */
/* -------------------------------------------------------------------- */

block_t*          block_CreateEmptyBlock       (const integer_t n, 
												const integer_t m, 
												const integer_t ku, 
												const integer_t kl, 
												blocktype_t type,
												blocksection_t section);

Error_t           block_InitializeToValue       ( block_t* B, const complex_t value );

Error_t           block_Print                   ( block_t* B, const char* msg);

Error_t           block_Deallocate              ( block_t* B);

Bool_t            block_AreEqual                ( block_t* A, block_t* B );

static Error_t    block_Transpose               ( block_t* B );

block_t*          block_ExtractBlock            ( block_t* B, blocksection_t section );



/* -------------------------------------------------------------------- */
/* .. Functions for reduced sytem assembly.                             */
/* -------------------------------------------------------------------- */
matrix_t*         matrix_CreateEmptyReducedSystem  (const integer_t TotalPartitions, 
													integer_t *n, 
													integer_t *ku, 
													integer_t *kl );

static integer_t* ComputeReducedSytemDimensions	   (integer_t partitions, 
													integer_t *ku, 
													integer_t *kl);

static Error_t    GetNnzAndRowsUpToPartition       (const integer_t TotalPartitions, 
													const integer_t CurrentPartition, 
													integer_t *ku, integer_t *kl, 
													integer_t *nnz, 
													integer_t *FirstBlockRow );

Error_t           matrix_AddBlockToReducedSystem   (const integer_t TotalPartitions,
													const integer_t CurrentPartition,
													integer_t          *n,
													integer_t          *ku,
													integer_t          *kl,
													matrix_t           *R,
													block_t            *B);

