#include "spike_algebra.h"

Error_t reorder_metis ( const integer_t n,
						const integer_t nnz,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij,
						integer_t* colperm )
{

};

Error_t reorder_fiedler( const integer_t n,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij,
						integer_t *restrict colperm,
						integer_t *restrict scale )
{

};

Error_t reorder_rcm (const integer_t n,
					integer_t *restrict colind,
					integer_t *restrict rowptr,
					complex_t *restrict aij,
					integer_t *restrict colperm)
{

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
	integer_t row, col, idx;
	*ku = 0;
	*kl = 0;

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
		fprintf(stderr, "\n\tProcessing residual for "_I_"-th RHS vector", rhs + 1);

		// set vector to zero
		memset((void*) Ax, 0, n * sizeof(complex_t));

		/* compute Ax - b using mkl_cspblas_?csrgemv*/
		CALL_LA_KERNEL(mkl_cspblas_,_PPREF_,csrgemv) ("N", &n, aij, rowptr, colind, &x[rhs * n], Ax);
		
		/* compute norm of b using cblas_?nrm2*/
		bnorm = nrm2(n, &b[rhs * n], 1 );

		///* subtracts b to Ax using cblas_?axpy */
		axpy(n, __nunit , &b[rhs * n], 1, Ax, 1 );

		///* compute the norm of Ax - b */
		absres = nrm2( n, Ax, 1 );


		fprintf(stderr, "\n\t\tNorm of rhs vector                    %E", bnorm);
		fprintf(stderr, "\n\t\tAbsolute residual                     %E", absres);
		fprintf(stderr, "\n\t\tRelative residual                     %E", absres / bnorm );
		fprintf(stderr, "\n");
	}

	/* clean up */
	spike_nullify( Ax );

	return (SPIKE_SUCCESS);
};


/*
	Instead of using matrix_t structure, here we use the argument list that
	most back ends support.
 */
 Error_t system_solve ( integer_t *restrict colind, // ja
						integer_t *restrict rowptr, // ia
						complex_t *restrict aij,
						complex_t *restrict x,
						complex_t *restrict b,
						integer_t  n,
						integer_t  nrhs)
{
	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	timer_t start_t;
	timer_t ordering_t;
	timer_t factor_t;
	timer_t solve_t;

	MKL_INT mtype = MTYPE_GEN_NOSYMM;  	/* Real unsymmetric matrix */
	void *pt[64];       				/* Pardiso control parameters. */
	MKL_INT iparm[64]; 					/* Pardiso control parameters. */
	MKL_INT maxfct, mnum, phase, error, msglvl;
	double ddum;          				/* Double dummy */
	MKL_INT idum;         				/* Integer dummy. */

	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */
	memset((void*) iparm, 0, 64 * sizeof(MKL_INT));

	iparm[0]  =  1;    /* No solver default */
	iparm[1]  =  2;    /* Fill-in reordering from METIS */
	iparm[3]  =  0;    /* No iterative-direct algorithm */
	iparm[4]  =  0;    /* No user fill-in reducing permutation */
	iparm[5]  =  0;    /* Write solution into x */
	iparm[6]  =  0;    /* Not in use */
	iparm[7]  =  2;    /* Max numbers of iterative refinement steps */
	iparm[8]  =  0;    /* Not in use */
	iparm[9]  = 13;    /* Perturb the pivot elements with 1E-13 */
	iparm[10] =  1;    /* Use nonsymmetric permutation and scaling MPS */
	iparm[11] =  0;    /* Conjugate transposed/transpose solve */
	iparm[12] =  1;    /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
	iparm[13] =  0;    /* Output: Number of perturbed pivots */
	iparm[14] =  0;    /* Not in use */
	iparm[15] =  0;    /* Not in use */
	iparm[16] =  0;    /* Not in use */
	iparm[17] = -1;    /* Output: Number of nonzeros in the factor LU, -1 shows it */
	iparm[18] = -1;    /* Output: Mflops for LU factorization, -1 shows it */
	iparm[19] =  0;    /* Output: Numbers of CG Iterations */
	iparm[34] =  1;    /* Pardiso use C-style indexing for ia and ja arrays */
	maxfct    =  1;    /* Maximum number of numerical factorizations. */
	mnum      =  1;    /* Which factorization to use. */
	msglvl    =  0;    /* Print statistical information in file */
	error     =  0;    /* Initialize error flag */

	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only */
	/* necessary for the FIRST call of the PARDISO solver. */
	/* -------------------------------------------------------------------- */
	memset((void*) pt, 0, 64 * sizeof(MKL_INT));

	/* -------------------------------------------------------------------- */
	/* .. Reordering and Symbolic Factorization. This step also allocates */
	/* all memory that is necessary for the factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 11;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if ( error != 0 )
	{
			printf ("\nERROR during symbolic factorization: %d", error);
			exit (1);
	}
	ordering_t = GetReferenceTime() - start_t;

	// printf ("\nReordering completed ... ");
	// printf ("\nNumber of nonzeros in factors = %d", iparm[17]);
	// printf ("\nNumber of factorization MFLOPS = %d", iparm[18]);
	
	// fprintf(stderr, "\n%s: Reordering and Symbolic Factorization time %.6lf", __FUNCTION__, ordering_t );

	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 22;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	if ( error != 0 )
	{
			printf ("\nERROR during numerical factorization: %d", error);
			exit (2);
	}
	factor_t = GetReferenceTime() - start_t;
	
	//fprintf(stderr, "\n%s: Numerical factorization time %.6lf", __FUNCTION__, factor_t );

	//printf ("\nFactorization completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Back substitution and iterative refinement. */
	/* -------------------------------------------------------------------- */
	fprintf (stderr, "\n\nSolving linear system...\n");

	start_t = GetReferenceTime();
	phase = 33;

	if ( x == NULL ) {
		PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
			&n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, b, b, &error);
	}
	else {
		PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
			&n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, b, x, &error);
	
	}
	if ( error != 0 )
	{
			printf ("\nERROR during solution: %d", error);
			exit (3);
	}
	solve_t = GetReferenceTime() - start_t;
	// fprintf(stderr, "\n%s: Solution time time %.6lf", __FUNCTION__, solve_t );

	/* -------------------------------------------------------------------- */
	/* .. Compute Residual. */
	/* -------------------------------------------------------------------- */
    ComputeResidualOfLinearSystem( colind, rowptr, aij, x, b, n, nrhs );

	/* -------------------------------------------------------------------- */
	/* .. Termination and release of memory. */
	/* -------------------------------------------------------------------- */
	phase = -1;           /* Release internal memory. */
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, &ddum, rowptr, colind, &idum, &nrhs,
					 iparm, &msglvl, &ddum, &ddum, &error);

	/* print some efficiency statistics */
	// fprintf(stderr, "\n%s: Factor to solve ratio: %.6f", __FUNCTION__, factor_t / solve_t );

	return (SPIKE_SUCCESS);
};


/*
	Instead of using matrix_t structure, here we use the argument list that
	most back ends support.

	This function factorizes the coefficient matrix, keeping the factors
	in memory for the later resolution of multiple RHS.
 */
 Error_t directSolver_Factorize(integer_t *restrict colind, // ja
								integer_t *restrict rowptr, // ia
								complex_t *restrict aij,
								integer_t  n,
								integer_t  nrhs,
								void *pt)
{
	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	timer_t start_t;
	timer_t ordering_t;
	timer_t factor_t;

	MKL_INT mtype = MTYPE_GEN_NOSYMM;       	/* Real unsymmetric matrix */
	MKL_INT iparm[64]; 							/* Pardiso control parameters. */
	MKL_INT maxfct, mnum, phase, error, msglvl;
	double ddum;          						/* Double dummy */
	MKL_INT idum;         						/* Integer dummy. */

	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */
	memset((void*) iparm, 0, 64 * sizeof(MKL_INT));

	iparm[0]  =  1;    /* No solver default */
	iparm[1]  =  2;    /* Fill-in reordering from METIS */
	iparm[3]  =  0;    /* No iterative-direct algorithm */
	iparm[4]  =  0;    /* No user fill-in reducing permutation */
	iparm[5]  =  0;    /* Write solution into x */
	iparm[6]  =  0;    /* Not in use */
	iparm[7]  =  2;    /* Max numbers of iterative refinement steps */
	iparm[8]  =  0;    /* Not in use */
	iparm[9]  = 13;    /* Perturb the pivot elements with 1E-13 */
	iparm[10] =  1;    /* Use nonsymmetric permutation and scaling MPS */
	iparm[11] =  0;    /* Conjugate transposed/transpose solve */
	iparm[12] =  1;    /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
	iparm[13] =  0;    /* Output: Number of perturbed pivots */
	iparm[14] =  0;    /* Not in use */
	iparm[15] =  0;    /* Not in use */
	iparm[16] =  0;    /* Not in use */
	iparm[17] = -1;    /* Output: Number of nonzeros in the factor LU, -1 shows it */
	iparm[18] = -1;    /* Output: Mflops for LU factorization, -1 shows it */
	iparm[19] =  0;    /* Output: Numbers of CG Iterations */
	iparm[34] =  1;    /* Pardiso use C-style indexing for ia and ja arrays */
	maxfct    =  1;    /* Maximum number of numerical factorizations. */
	mnum      =  1;    /* Which factorization to use. */
	msglvl    =  0;    /* Print statistical information in file */
	error     =  0;    /* Initialize error flag */

	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only       */
	/* necessary for the FIRST call of the PARDISO solver.                  */
	/* -------------------------------------------------------------------- */
	memset((void*) pt, 0, 64 * sizeof(MKL_INT));

	/* -------------------------------------------------------------------- */
	/* .. Reordering and Symbolic Factorization. This step also allocates */
	/* all memory that is necessary for the factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 11;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
			&n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if ( error != 0 )
	{
			printf ("\nERROR during symbolic factorization: %d", error);
			exit (1);
	}
	ordering_t = GetReferenceTime() - start_t;

	// printf ("\nReordering completed ... ");
	// printf ("\nNumber of nonzeros in factors = %d", iparm[17]);
	// printf ("\nNumber of factorization MFLOPS = %d", iparm[18]);
	// fprintf(stderr, "\n%s: Reordering and Symbolic Factorization time %.6lf", __FUNCTION__, ordering_t );


	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 22;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
			&n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	if ( error != 0 )
	{
			printf ("\nERROR during numerical factorization: %d", error);
			exit (2);
	}
	factor_t = GetReferenceTime() - start_t;
	
	//printf ("\nFactorization completed ... ");
	// fprintf(stderr, "\n%s: Numerical factorization time %.6lf\n", __FUNCTION__, factor_t );


	return (SPIKE_SUCCESS);
};

/*
	Instead of using matrix_t structure, here we use the argument list that
	most back ends support.

	This function factorizes uses a prevously computed factorization of the
	coefficient matrix to compute the solution for a given RHS vector.
 */
 Error_t directSolver_ApplyFactorToRHS  ( integer_t *restrict colind, // ja
						integer_t *restrict rowptr, // ia
						complex_t *restrict aij,
						complex_t *restrict x,
						complex_t *restrict b,
						integer_t  n,
						integer_t  nrhs,
						void *pt)
{
/* -------------------------------------------------------------------- */
/* .. Local variables. */
/* -------------------------------------------------------------------- */
	timer_t start_t;
	timer_t solve_t;

	MKL_INT mtype = MTYPE_GEN_NOSYMM;  /* Real unsymmetric matrix */
	MKL_INT maxfct, mnum, phase, error, msglvl;
	MKL_INT iparm[64]; /* Pardiso control parameters. */
	double ddum;          /* Double dummy */
	MKL_INT idum;         /* Integer dummy. */

/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
	memset((void*) iparm, 0, 64 * sizeof(MKL_INT));

	iparm[0]  =  1;    /* No solver default */
	iparm[1]  =  2;    /* Fill-in reordering from METIS */
	iparm[3]  =  0;    /* No iterative-direct algorithm */
	iparm[4]  =  0;    /* No user fill-in reducing permutation */
	iparm[5]  =  0;    /* Write solution into x */
	iparm[6]  =  0;    /* Not in use */
	iparm[7]  =  2;    /* Max numbers of iterative refinement steps */
	iparm[8]  =  0;    /* Not in use */
	iparm[9]  = 13;    /* Perturb the pivot elements with 1E-13 */
	iparm[10] =  1;    /* Use nonsymmetric permutation and scaling MPS */
	iparm[11] =  0;    /* Conjugate transposed/transpose solve */
	iparm[12] =  1;    /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
	iparm[13] =  0;    /* Output: Number of perturbed pivots */
	iparm[14] =  0;    /* Not in use */
	iparm[15] =  0;    /* Not in use */
	iparm[16] =  0;    /* Not in use */
	iparm[17] = -1;    /* Output: Number of nonzeros in the factor LU, -1 shows it */
	iparm[18] = -1;    /* Output: Mflops for LU factorization, -1 shows it */
	iparm[19] =  0;    /* Output: Numbers of CG Iterations */
	iparm[34] =  1;    /* Pardiso use C-style indexing for ia and ja arrays */
	maxfct    =  1;    /* Maximum number of numerical factorizations. */
	mnum      =  1;    /* Which factorization to use. */
	msglvl    =  0;    /* Print statistical information in file */
	error     =  0;    /* Initialize error flag */

/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
	// fprintf (stderr, "\n\nBack substitution and iterative refinement...\n");

	start_t = GetReferenceTime();
	phase = 33;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, b, x, &error);
	if ( error != 0 )
	{
			printf ("\nERROR during solution: %d", error);
			exit (3);
	}
	solve_t = GetReferenceTime() - start_t;
	
	// fprintf(stderr, "\n%s: Solution time time %.6lf", __FUNCTION__, solve_t );

/* -------------------------------------------------------------------- */
/* .. Compute Residual. */
/* -------------------------------------------------------------------- */
    
    // ComputeResidualOfLinearSystem( colind, rowptr, aij, x, b, n, nrhs );

/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */

	return (SPIKE_SUCCESS);
};

Error_t directSolver_CleanUp(integer_t *restrict colind, // ja
						integer_t *restrict rowptr, // ia
						complex_t *restrict aij,
						complex_t *restrict x,
						complex_t *restrict b,
						integer_t  n,
						integer_t  nrhs,
						void *pt)

{

/* -------------------------------------------------------------------- */
/* .. Local variables. */
/* -------------------------------------------------------------------- */
	MKL_INT mtype = MTYPE_GEN_NOSYMM;       /* Real unsymmetric matrix */
	MKL_INT iparm[64]; /* Pardiso control parameters. */
	MKL_INT maxfct, mnum, phase, error, msglvl;
	double ddum;          /* Double dummy */
	MKL_INT idum;         /* Integer dummy. */

/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
	phase = -1;           /* Release internal memory. */
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
			&n, &ddum, rowptr, colind, &idum, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);

	return (SPIKE_SUCCESS);
}


void symbolic_factorization ( void )
{

};

/*
	Performs the matrix-vector multiplication operation: b = Ax
*/
Error_t spmv( const integer_t n, const integer_t m, complex_t* aij, integer_t *colind, integer_t *rowptr, complex_t *x, complex_t *b)
{

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

		al(cblas_,_PPREF_,gemm)( Layout, transa, transb,
			m,    						/* m - number of rows of A    */
			n, 							/* n - number of columns of B */
			k,    						/* k - number of columns of A */
			alpha,						/* alpha                      */
			a, 							/* A block                    */
			lda,    					/* lda - first dimension of A */
			b, 							/* B block                    */
			ldb,    					/* ldb - first dimension of B */
			beta,	 					/* beta                       */
			c,							/* C block                    */
			ldc );		 				/* ldc - first dimension of C */

	#else
		CBLAS_LAYOUT Layout = (layout      == _ROWMAJOR_ ) ? CblasRowMajor : CblasColMajor;
		CBLAS_LAYOUT transa = (transpose_a == _TRANSPOSE_) ? CblasTrans    : CblasNoTrans;
		CBLAS_LAYOUT transb = (transpose_b == _TRANSPOSE_) ? CblasTrans    : CblasNoTrans;

		/* check for conjugate tranpose case: */
		transa = (transpose_a == _CONJTRANSPOSE_) ? CblasConjTrans    : CblasNoTrans;
		transb = (transpose_b == _CONJTRANSPOSE_) ? CblasConjTrans    : CblasNoTrans;

		CALL_LA_KERNEL(cblas_,_PPREF_,gemm)( Layout, transa, transb,
			m,    						/* m - number of rows of A    */
			n, 							/* n - number of columns of B */
			k,    						/* k - number of columns of A */
			(const void*) &alpha,		/* alpha                      */
			a, 							/* A block                    */
			lda,    					/* lda - first dimension of A */
			b, 							/* B block                    */
			ldb,    					/* ldb - first dimension of B */
			(const void*) &beta,	 	/* beta                       */
			c,							/* C block                    */
			ldc );		 				/* ldc - first dimension of C */

	#endif
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
};