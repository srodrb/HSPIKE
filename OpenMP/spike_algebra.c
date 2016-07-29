#include "spike_algebra.h"

Error_t reorder_metis ( const integer_t n,
						const integer_t nnz,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij,
						integer_t* colperm )
{
	return (SPIKE_SUCCESS);
};

Error_t reorder_fiedler( const integer_t n,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij,
						integer_t *restrict colperm,
						integer_t *restrict scale )
{
	return (SPIKE_SUCCESS);
};

Error_t reorder_rcm (const integer_t n,
					integer_t *restrict colind,
					integer_t *restrict rowptr,
					complex_t *restrict aij,
					integer_t *restrict colperm)
{
	return (SPIKE_SUCCESS);
};

Error_t matrix_ComputeBandwidth(const integer_t n,
								integer_t *restrict colind,
								integer_t *restrict rowptr,
								complex_t *restrict aij,
								integer_t *ku,
								integer_t *kl )
{
	// TODO: es posible calcular el bw mas rapidamente accediendo
	// solamente a las posiciones extremas de las filas.
	spike_timer_t tstart, tend;
	integer_t row, col, idx;
	*ku = 0;
	*kl = 0;

	/* initialize timer */
	tstart = GetReferenceTime();


	for(row = 0; row < n; row++)
	{
		for(idx = rowptr[row]; idx < rowptr[row+1]; idx++)
		{
			col = colind[idx];

			*ku = ((row - col) < *ku) ? (row - col) : *ku;
			*kl = ((col - row) < *kl) ? (col - row) : *kl;
		}
	}

	*ku = abs(*ku);
	*kl = abs(*kl);

	tend = GetReferenceTime() - tstart;

	fprintf(stderr, "\n%s: took %.6lf seconds", __FUNCTION__, tend);

	return (SPIKE_SUCCESS);
};

Error_t ComputeResidualOfLinearSystem ( integer_t *restrict colind,
										integer_t *restrict rowptr,
										complex_t *restrict aij,
										complex_t *restrict x,
										complex_t *restrict b,
										integer_t n,
										integer_t nrhs )
{
	//
	// TODO add support for complex norm of vectors
	// https://software.intel.com/es-es/node/520747#64961E94-92D0-4671-90E6-86995E259A85
	// cblas_?cabs1
	//
	
	real_t absres = 0.0; /* absolute residual */
	real_t relres = 0.0; /* relative residual */
	real_t bnorm  = 0.0;  /* norm of b        */

	complex_t *Ax = (complex_t*) spike_malloc( ALIGN_COMPLEX, n, sizeof(complex_t));

	for( integer_t rhs = 0; rhs < nrhs; rhs++){
		if ( rhs < 5 ){
			fprintf(stderr, "\n\tProcessing residual for "_I_"-th RHS vector", rhs + 1);

			/* set buffer array to zero */
			memset((void*) Ax, 0, n * sizeof(complex_t));

			/* compute Ax - b using mkl_cspblas_?csrgemv*/
			CALL_LA_KERNEL(mkl_cspblas_,_PPREF_,csrgemv) ("N", &n, aij, rowptr, colind, &x[rhs * n], Ax);
			
			/* compute norm of b using cblas_?nrm2*/
			bnorm = nrm2(n, &b[rhs * n], 1 );

			/* subtracts b to Ax using cblas_?axpy */
			axpy(n, __nunit , &b[rhs * n], 1, Ax, 1 );

			/* compute the norm of Ax - b */
			absres = nrm2( n, Ax, 1 );

			/* Show norms of the residuals */
			fprintf(stderr, "\n\t\tNorm of rhs vector                    %E", bnorm);
			fprintf(stderr, "\n\t\tAbsolute residual                     %E", absres);
			fprintf(stderr, "\n\t\tRelative residual                     %E", absres / bnorm );
			fprintf(stderr, "\n");
		}
	}

	/* clean up */
	spike_nullify( Ax );

	return (SPIKE_SUCCESS);
};

void symbolic_factorization ( void )
{

};

/*
	Performs the matrix-vector multiplication operation: b = Ax
*/
Error_t spmv( const integer_t n, const integer_t m, complex_t* aij, integer_t *colind, integer_t *rowptr, complex_t *x, complex_t *b)
{
	return (SPIKE_SUCCESS);
};

/*
	Performs general matrix-matrix multiplication.
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
				const integer_t ldc)
{
	/* Intel's MKL back-end */

	#ifndef _COMPLEX_ARITHMETIC_

		CBLAS_LAYOUT Layout = (layout      == _ROWMAJOR_ ) ? CblasRowMajor : CblasColMajor;
		CBLAS_LAYOUT transa = (transpose_a == _TRANSPOSE_) ? CblasTrans    : CblasNoTrans;
		CBLAS_LAYOUT transb = (transpose_b == _TRANSPOSE_) ? CblasTrans    : CblasNoTrans;

		CALL_LA_KERNEL(cblas_,_PPREF_,gemm)( Layout, transa, transb,
			m,    						/* m - number of rows of A    */
			n, 								/* n - number of columns of B */
			k,    						/* k - number of columns of A */
			alpha,						/* alpha                      */
			a, 								/* A block                    */
			lda,    					/* lda - first dimension of A */
			b, 								/* B block                    */
			ldb,    					/* ldb - first dimension of B */
			beta,	 						/* beta                       */
			c,								/* C block                    */
			ldc );		 				/* ldc - first dimension of C */

	#else
		CBLAS_LAYOUT Layout = (layout      == _ROWMAJOR_ ) ? CblasRowMajor : CblasColMajor;
		CBLAS_LAYOUT transa = (transpose_a == _TRANSPOSE_) ? CblasTrans    : CblasNoTrans;
		CBLAS_LAYOUT transb = (transpose_b == _TRANSPOSE_) ? CblasTrans    : CblasNoTrans;

		/* check for conjugate tranpose case: */
		transa = (transpose_a == _CONJTRANSPOSE_) ? CblasConjTrans    : CblasNoTrans;
		transb = (transpose_b == _CONJTRANSPOSE_) ? CblasConjTrans    : CblasNoTrans;

		CALL_LA_KERNEL(cblas_,_PPREF_,gemm)( Layout, transa, transb,
			m,    									/* m - number of rows of A    */
			n, 											/* n - number of columns of B */
			k,    									/* k - number of columns of A */
			(const void*) &alpha,		/* alpha                      */
			a, 											/* A block                    */
			lda,    								/* lda - first dimension of A */
			b, 											/* B block                    */
			ldb,    								/* ldb - first dimension of B */
			(const void*) &beta,	 	/* beta                       */
			c,											/* C block                    */
			ldc );		 							/* ldc - first dimension of C */

	#endif

	return (SPIKE_SUCCESS);
}

real_t nrm2(const integer_t n, complex_t *restrict x, const integer_t incx)
{
	real_t norm = 0.0;

	#if defined (_DATATYPE_Z_) // double complex
		norm = cblas_dznrm2 (n, (const void*) x, incx);
	 
	#elif defined (_DATATYPE_C_) // complex float
		norm = cblas_scnrm2 (n, (const void*) x, incx);

	#elif defined (_DATATYPE_D_) // double precision float
		norm = cblas_dnrm2 (n, (const double*) x, incx);

	#else // single precision float
		norm = cblas_snrm2 (n, (const float*) x, incx);
	#endif

	return (norm);
};

Error_t axpy(const integer_t n, const complex_t alpha, complex_t* restrict x, const integer_t incx, complex_t *restrict y, const integer_t incy)
{
	#ifndef _COMPLEX_ARITHMETIC_
		CALL_LA_KERNEL(cblas_,_PPREF_,axpy) (n, alpha, x, incx, y, incy);
	#else
		CALL_LA_KERNEL(cblas_,_PPREF_,axpy) (n, (const void*) &alpha, (const void*) x, incx, (void*) y, incy);
	#endif

	return (SPIKE_SUCCESS);
};