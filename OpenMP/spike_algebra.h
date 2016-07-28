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

 	/* superlu interface */
 	#include "slu_mt_ddefs.h"
 	#include "slu_mt_util.h"

 	/* INTEL MKL interface */
	#include "mkl_pardiso.h"
	#include "mkl_types.h"
	#include "mkl_spblas.h"
 	#include "mkl.h"
	#include "mkl_cblas.h"

 	/*
 		Depending on the back-end, the nature of the coeffient matrix is specified
 		differently. In general, each backend has a predefined list of integer values
 		to identify each type of matrix.

 		Here we support some of them.
 	 */
 	#ifdef __INTEL_MKL__
		#ifndef _COMPLEX_ARITHMETIC_
			#define MTYPE_STRUC_SYMM    1	/* Real and structurally symmetric         	*/
			#define MTYPE_POSDEF  	    2	/* Real and symmetric positive definite    	*/
			#define MTYPE_SYMM_INDEF   -2	/* Real and symmetric indefinite           	*/
			#define MTYPE_GEN_NOSYMM   11	/* Real and nonsymmetric matrix 			*/
		#else
			#define MTYPE_STRUC_SYMM    3	/* Complex and structurally symmetric      	*/
			#define MTYPE_HERM_POSDEF   4	/* Complex and Hermitian positive definite 	*/
			#define MTYPE_HERM_INDEF   -4	/* Complex and Hermitian indefinite 		*/
			#define MTYPE_SYMM          6	/* Complex and symmetric matrix 			*/
			#define MTYPE_GEN_NOSYMM   13	/* Complex and nonsymmetric matrix 			*/
	 	#endif
 	#endif


 	typedef struct {
 		
		superlumt_options_t 	superlumt_options;
		superlu_memusage_t 		superlu_memusage;
		// Gstat_t  Gstat;
		
		integer_t nprocs;
    	fact_t fact;
    	trans_t trans;
    	yes_no_t refact;
    	yes_no_t usepr;
    	equed_t equed;
    	void *work;
    	integer_t info;
    	integer_t lwork;
    	integer_t panel_size;
    	integer_t relax;
    	integer_t permc_spec;
    	real_t u;
    	real_t drop_tol;
    	real_t rpg;
		real_t recip_pivot_growth; // ???
		real_t rcond;

		integer_t n;    /* matrix dimension                    */
    	integer_t ldx;	/* matrix leading dimension            */
		integer_t nnz;  /* nnz elements in A                   */
		integer_t nrhs; /* number of columns of x and b arrays */
		 

		SuperMatrix 			A; 
		SuperMatrix 			L;
		SuperMatrix 			U;
		SuperMatrix 			B;
		SuperMatrix 			X;

		integer_t 				*perm_c;
		integer_t 				*perm_r;
		real_t 					*R;
		real_t 					*C;
		real_t                  *berr;
		real_t                  *ferr;

		integer_t *etree; 			/* elimination tree */
		integer_t *colcnt_h; 		/* column count */
		integer_t *part_super_h; 	/* supernode partition for the Householder matrix */

 	} DirectSolverHander_t;


 	/*
 		These macros are intended to build the name of the linear algebra backends
 		according to the datatype being used.
 	 */
	#define CAT_I(a,b,c) a##b##c
	#define CAT(a,b,c)   CAT_I(a,b,c)
	#define CALL_LA_KERNEL(lib,prec,call) CAT(lib,prec,call)

 	typedef enum { _TRANSPOSE_, _CONJTRANSPOSE_, _NOTRANSPOSE_} transpose_t;
 	typedef enum { _ROWMAJOR_, _COLMAJOR_ }   memlayout_t;

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

	/*
		Interface for general matrix-matrix multiplication
	*/
	Error_t gemm   (const memlayout_t layout, 
					const transpose_t transpose_a, 
					const transpose_t transpose_b, 
					const integer_t m, 
					const integer_t n, 
					const integer_t k,
					const complex_t alpha,
					complex_t *restrict a,
					const integer_t lda,
					complex_t *restrict b,
					const integer_t ldb,
					const complex_t beta,
					complex_t *restrict c,
					const integer_t ldc);

	real_t nrm2(const integer_t n, complex_t *restrict x, const integer_t incx);

	Error_t axpy(const integer_t n, const complex_t alpha, complex_t* restrict x, const integer_t incx, complex_t *restrict y, const integer_t incy);

void superlu_solve (const integer_t n, 
                    const integer_t nnz,
                    const integer_t nrhs,
                    integer_t *restrict colind,
                    integer_t *restrict rowptr,
                    complex_t *restrict aij,
                    complex_t *restrict x,
                    complex_t *restrict b);

#endif /* end of _SPIKE_ALGEBRA_H_ definition */
