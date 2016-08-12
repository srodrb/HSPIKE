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
 *  Syntax of the function: <affected abstraction>_<function name>
 * 
 *
 * =====================================================================================
 */
#ifndef _SPIKE_MATRIX_H_
	#define _SPIKE_MATRIX_H_

	#include "spike_memory.h"
	#include "spike_common.h"
	#include "spike_datatypes.h"
 	#include "spike_algebra.h"
	#include "mkl.h"
	#include <string.h>

	#define _MAX_PRINT_DIMENSION_ 50
	#define _MAX_PRINT_RHS_        3

 	typedef enum { _C_BLOCK_, _DIAG_BLOCK_, _B_BLOCK_ } matrixtype_t;

	/* -------------------------------------------------------------------- */
	/* .. Dense block structure.                                            */
	/* -------------------------------------------------------------------- */
	typedef struct
	{
		integer_t n;
		uLong_t   nnz;

		integer_t ku;
		integer_t kl;

		matrixtype_t type;

		integer_t* colind; // TODO attribute aligned
		integer_t* rowptr; // TODO attribute aligned
		complex_t* aij;    // TODO attribute aligned

	} matrix_t;

	/* -------------------------------------------------------------------- */
	/* .. Dense block structure.                                            */
	/* -------------------------------------------------------------------- */

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


	integer_t*        vector_LoadPermutationArray   (const integer_t n, const char* filename);

	complex_t*        vector_LoadRHS                (const integer_t n, const char* filename );

	Error_t           matrix_ExportBinary           ( matrix_t* M, const char* filename );

	matrix_t*         matrix_CreateEmptyMatrix      (const integer_t n, const uLong_t nnz );

	matrix_t*         matrix_CreateFromComponents  (const integer_t n, 
													const uLong_t nnz, 
													integer_t *restrict colind, 
													integer_t *restrict rowptr, 
													complex_t *restrict aij);


	Error_t           matrix_Deallocate             (matrix_t* M );

	Bool_t            matrix_AreEqual               (matrix_t* A, matrix_t* B );

	Error_t           matrix_PrintAsSparse          (matrix_t* M, const char* msg);

	Error_t           matrix_PrintAsDense           (matrix_t* A, const char* msg);

	block_t*          matrix_ExtractBlock 		   (matrix_t* M,
									            	const integer_t r0,
									            	const integer_t rf,
									            	const integer_t c0,
									            	const integer_t cf,
									            	blocktype_t type);

	Error_t        matrix_ExtractBlock_blocking  (  matrix_t* M,
													block_t* B,
													const integer_t r0,
													const integer_t rf,
													const integer_t c0,
													const integer_t cf,
													blocktype_t type );

	matrix_t*         matrix_ExtractMatrix 		   (matrix_t* M,
													const integer_t r0,
													const integer_t rf,
													const integer_t c0,
													const integer_t cf,
													matrixtype_t type );
Error_t block_BuildBlockFromMatrix_blocking (  matrix_t* M,
								block_t* B,
								const integer_t r0,
								const integer_t rf,
								const integer_t c0,
								const integer_t cf,
								blocktype_t type );

Error_t block_ExtractTip_blocking_mpi ( block_t *dst,
									block_t *src, 
									const integer_t c0,
									const integer_t cf,
									const integer_t colOffset,
									blocksection_t section,
									memlayout_t layout );

	/* -------------------------------------------------------------------- */
	/* .. Functions affecting block structures.                             */
	/* -------------------------------------------------------------------- */

	block_t*          block_CreateEmptyBlock       (const integer_t n, 
													const integer_t m, 
													const integer_t ku, 
													const integer_t kl, 
													blocktype_t type,
													blocksection_t section);

	block_t* 		  block_CreateFromComponents   (const integer_t n,
													const integer_t m,
													complex_t *restrict Bij);

	block_t* 		  block_BuildBlockFromMatrix  ( matrix_t *M,
													blocktype_t type,
													const integer_t n,
													const integer_t m,
													const integer_t ku,
													const integer_t kl );
	
	Error_t           block_InitializeToValue       ( block_t* B, const complex_t value );

	Error_t           block_Print                   ( block_t* B, const char* msg);

	Error_t           block_Deallocate              ( block_t* B);

	Bool_t            block_AreEqual                ( block_t* A, block_t* B );

	Error_t    block_Transpose               ( block_t* B );

	Error_t block_Transpose_blocking( complex_t* aij, const integer_t n, const integer_t m );

	block_t*          block_ExtractTip              ( block_t* B, blocksection_t section, memlayout_t layout );

	Error_t           block_ExtractTip_blocking   ( block_t *dst,
													block_t *src, 
													const integer_t c0,
													const integer_t cf,
													blocksection_t section,
													memlayout_t layout );

	block_t*          block_ExtractBlock            (block_t* B, 
													const integer_t n0,
													const integer_t nf);

	Error_t           block_ExtractBlock_blocking ( block_t *dst, 
													block_t *src, 
													const integer_t n0, 
													const integer_t nf,
													const integer_t c0,
													const integer_t cf );

	Error_t           block_SetBandwidthValues      (block_t* B,
													const integer_t ku,
													const integer_t kl);

	block_t*          block_CreateReducedRHS       (const integer_t TotalPartitions,
													integer_t *ku,
													integer_t *kl,
													const integer_t nrhs);

	Error_t           block_AddTipTOReducedRHS      (const integer_t CurrentPartition,
													integer_t          *ku,
													integer_t          *kl,
													block_t            *RHS,
													block_t            *B);

	Error_t           matrix_AddTipToReducedMatrix_blocking (const integer_t TotalPartitions,
													const integer_t CurrentPartition,
													const integer_t    c0,
													const integer_t    cf,
													integer_t          *n,
													integer_t          *ku,
													integer_t          *kl,
													matrix_t           *R,
													block_t            *B);


	Error_t           block_AddBlockToRHS          (block_t* x, block_t* xi,
													const integer_t n0,
													const integer_t nf);

	Error_t block_AddTipTOReducedRHS_blocking   	(const integer_t CurrentPartition,
													const integer_t     c0,
													const integer_t     cf,
													integer_t          *ku,
													integer_t          *kl,
													block_t            *RHS,
													block_t            *B);


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
														integer_t *ku,
														integer_t *kl, 
														uLong_t *nnz, 
														integer_t *FirstBlockRow );

	Error_t           matrix_AddTipToReducedMatrix   (const integer_t TotalPartitions,
														const integer_t CurrentPartition,
														integer_t          *n,
														integer_t          *ku,
														integer_t          *kl,
														matrix_t           *R,
														block_t            *B);

	Error_t           reduced_PrintAsDense           (matrix_t *R, block_t *X, block_t *Y, const char* msg);

/* end of _SPIKE_MATRIX_H_ definition */
#endif
