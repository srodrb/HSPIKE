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

 	#include "spike_datatypes.h"
 	#include "spike_common.h"
 	#include "spike_memory.h"

#ifdef __INTEL_COMPILER
	#include "mkl_pardiso.h"
	#include "mkl_types.h"
	#include "mkl_spblas.h"
 	#include "mkl.h"
#endif

	/*
	 * Uses metis to compute a permutation that minizes the fill-in on the LU factors of A
	 */
	Error_t reorder_metis(  const integer_t n,
							const integer_t nnz,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							complex_t *restrict aij,
							integer_t *restrict colperm);

	/*
	 * Computes the Fielder permutation of the Laplacian graph of A.
	 */
	Error_t reorder_fiedler(const integer_t n,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							complex_t *restrict aij,
							integer_t *restrict colperm,
							integer_t *restrict scale);

	/*
	 * Computes a permutation that reduces the bandwidth of the matrix A using
	 * the Reverse Cuthill Mckee algorithm.
	 *
	 * This algorithm is implemented in linear time.
	 */
	Error_t reorder_rcm(const integer_t n,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij,
						integer_t *restrict colperm);

	/*
	 * Computes the upper and lower bandwidth of the matrix A.
	 */
	Error_t matrix_ComputeBandwidth(const integer_t n,
									integer_t *restrict colind,
									integer_t *restrict rowptr,
									complex_t *restrict aij,
									integer_t *ku,
									integer_t *kl);

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
	void symbolic_factorization ( void );


#endif /* end of _SPIKE_ALGEBRA_H_ definition */
