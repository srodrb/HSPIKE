/*
 * =====================================================================================
 *
 *       Filename:  spike_algebra.h
 *
 *    Description:  Here are the interfaces for different linear algebra backends.
 *
 *        Version:  1.0
 *        Created:  22/06/16 10:51:25
 *       Revision:  none
 *       Compiler:  icc/nvcc/others
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#ifndef _SPIKE_ALGEBRA_H_
	#define _SPIKE_ALGEBRA_H_

	#include "spike_matrix.h"
	
#ifdef __INTEL_COMPILER
	#include "mkl_pardiso.h"
	#include "mkl_types.h"
	#include "mkl_spblas.h"
 	#include "mkl.h"
#endif

	/*
	 * Uses metis to compute a permutation that minizes the fill-in on the LU factors of A
	 */
	void reorder_metis( matrix_t* A, integer_t* colperm );

	/*
	 * Computes the Fielder permutation of the Laplacian graph of A.
	 */
	void reorder_fieldler( matrix_t* A, integer_t* colperm, integer_t* scale );

	/*
	 * Computes a permutation that reduces the bandwidth of the matrix A using
	 * the Reverse Cuthill Mckee algorithm.
	 *
	 * This algorithm is implemented in linear time.
	 */
	void reorder_rcm ( matrix_t* A, integer_t* colperm );

	/*
	 * Solves the sparse linear system Ax=b, where A is an sparse CSR matrix.
	 * if x is NULL, then the solution of the system is stored on B.
	 */
	 Error_t system_solve ( integer_t *restrict colind, // ja
	 										    integer_t *restrict rowptr, // ia
	 										    complex_t *restrict aij,
	 										    complex_t *restrict x,
	 										    complex_t *restrict b,
	 										    integer_t  n,
	 										    integer_t  nrhs);
	 
	 Error_t directSolver_Factorize(integer_t *restrict colind, // ja
									integer_t *restrict rowptr, // ia
									complex_t *restrict aij,
									integer_t  n,
									integer_t  nrhs,
									void *pt);

	 Error_t directSolver_ApplyFactorToRHS (integer_t *restrict colind, // ja
											integer_t *restrict rowptr, // ia
											complex_t *restrict aij,
											complex_t *restrict x,
											complex_t *restrict b,
											integer_t  n,
											integer_t  nrhs,
											void *pt);

	//Error_t directSolver_CleanUp (void *pt);

	Error_t directSolver_CleanUp(integer_t *restrict colind, // ja
						integer_t *restrict rowptr, // ia
						complex_t *restrict aij,
						complex_t *restrict x,
						complex_t *restrict b,
						integer_t  n,
						integer_t  nrhs,
						void *pt);



	 Error_t ComputeResidualOfLinearSystem (  integer_t *restrict colind,
										integer_t *restrict rowptr,
										complex_t *restrict aij,
										complex_t *restrict x,
										complex_t *restrict b,
										integer_t n,
										integer_t nrhs);

	/*
	 * Custom symbolic factorization routine used on the strategy design phase
	 */
	void symbolic_factorization ( matrix_t* A );

	/*
	 * Computes the upper and lower bandwidth of the matrix A.
	 */
	Error_t matrix_ComputeBandwidth( matrix_t* A );

#endif /* end of _SPIKE_ALGEBRA_H_ definition */
