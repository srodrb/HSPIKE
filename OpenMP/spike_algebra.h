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

 	/* INTEL MKL interface */
	#include "mkl_types.h"
	#include "mkl_spblas.h"
 	#include "mkl.h"
	#include "mkl_cblas.h"


 	/* The backend inclues also the definition of datatypes */
 	/* and other common headers                             */
 	#if defined (_PARDISO_BACKEND_)
 		#include "spike_pardiso.h"
 	#elif defined (_SUPERLU_BACKEND_)
 		#include "spike_superlu.h"
 	#elif defined (_NVIDIA_BACKEND_)
 		#include "spike_cuda.h"
 	#endif

	/* just for the host solver.. */
     /* Pardiso interface */
	#ifndef _PARDISO_BACKEND_ 
	    #include "mkl_pardiso.h"

	    #ifndef _COMPLEX_ARITHMETIC_
	        #define MTYPE_GEN_NOSYMM   11   /* Real and nonsymmetric matrix             */
	    #else
	        #define MTYPE_GEN_NOSYMM   13   /* Complex and nonsymmetric matrix          */
	    #endif
 	#endif

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
							const uLong_t nnz,
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
		Permutes the vector v according to the permutation vector
		p (i.e., performs v(p) Matlab's operation).
		The result is stored in p at exit.

		n (in) numer of columns of v
		m (in) number of rows of m
		p (in) integer array, dimension (n,1)
		v (in/out) vector to permute, dimension (n,m)

		The routine assumes that v is stored in Row-Major form.
	 */
	Error_t permute_vector( const integer_t flag,
							const integer_t n,
							const integer_t m,
							integer_t *restrict p,
							complex_t *restrict v );

	/*
	 * Computes the upper and lower bandwidth of the matrix A.
	 */
	Error_t matrix_ComputeBandwidth(const integer_t n,
									integer_t *restrict colind,
									integer_t *restrict rowptr,
									complex_t *restrict aij,
									integer_t *ku,
									integer_t *kl);

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

	/* host solver for the reduced system */
	 Error_t directSolver_Host_Solve (integer_t n,
 							integer_t nnz,
 							integer_t nrhs,
 							integer_t *restrict colind, // ja
							integer_t *restrict rowptr, // ia
							complex_t *restrict aij,
							complex_t *restrict x,
							complex_t *restrict b);


#endif /* end of _SPIKE_ALGEBRA_H_ definition */
