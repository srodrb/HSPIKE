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
	spike_timer_t start_t;
	spike_timer_t ordering_t;
	spike_timer_t factor_t;
	spike_timer_t solve_t;

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
	iparm[7]  =  3;    /* Max numbers of iterative refinement steps */
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
			fprintf (stderr, "\nERROR during symbolic factorization: %d", error);
			exit (1);
	}
	ordering_t = GetReferenceTime() - start_t;

	fprintf(stderr, "\n%s: Reordering completed ... ", __FUNCTION__);
	fprintf(stderr, "\n%s: Number of nonzeros in factors = %d", __FUNCTION__, iparm[17]);
	fprintf(stderr, "\n%s: Number of factorization MFLOPS = %d", __FUNCTION__, iparm[18]);
	fprintf(stderr, "\n%s: Reordering and Symbolic Factorization time %.6lf", __FUNCTION__, ordering_t );

	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 22;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	if ( error != 0 )
	{
			fprintf (stderr,"\nERROR during numerical factorization: %d", error);
			exit (2);
	}
	factor_t = GetReferenceTime() - start_t;
	
	fprintf (stderr, "\n%s: Factorization completed ... ", __FUNCTION__);
	fprintf (stderr, "\n%s: Numerical factorization time %.6lf", __FUNCTION__, factor_t );

	/* -------------------------------------------------------------------- */
	/* .. Back substitution and iterative refinement. */
	/* -------------------------------------------------------------------- */
	fprintf (stderr, "\n\n%s: Solving linear system...\n", __FUNCTION__ );

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
			fprintf (stderr,"\nERROR during solution: %d", error);
			exit (3);
	}
	solve_t = GetReferenceTime() - start_t;
	
	fprintf(stderr, "\n%s: Solution time time %.6lf", __FUNCTION__, solve_t );

	/* -------------------------------------------------------------------- */
	/* .. Compute Residual. */
	/* -------------------------------------------------------------------- */
    // ComputeResidualOfLinearSystem( colind, rowptr, aij, x, b, n, nrhs );

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
	spike_timer_t start_t;
	spike_timer_t ordering_t;
	spike_timer_t factor_t;

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
	iparm[1]  =  1;    /* Fill-in reordering from METIS */
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
			fprintf (stderr, "\nERROR during symbolic factorization: %d", error);
			exit (1);
	}
	ordering_t = GetReferenceTime() - start_t;

	fprintf(stderr, "\n%s: Reordering completed ... ", __FUNCTION__);
	fprintf(stderr, "\n%s: Number of nonzeros in factors = %d", __FUNCTION__, iparm[17]);
	fprintf(stderr, "\n%s: Number of factorization MFLOPS = %d", __FUNCTION__, iparm[18]);
	fprintf(stderr, "\n%s: Reordering and Symbolic Factorization time %.6lf", __FUNCTION__, ordering_t );


	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 22;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
			&n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	if ( error != 0 )
	{
			fprintf (stderr, "\nERROR during numerical factorization: %d", error);
			exit (2);
	}
	factor_t = GetReferenceTime() - start_t;
	
	fprintf(stderr, "\n%s: Factorization completed ... ", __FUNCTION__);
	fprintf(stderr, "\n%s: Numerical factorization time %.6lf\n", __FUNCTION__, factor_t );


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
	spike_timer_t start_t;
	spike_timer_t solve_t;

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
	iparm[1]  =  1;    /* Fill-in reordering from METIS */
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
	fprintf (stderr, "\n\n%s: Back substitution and iterative refinement...\n", __FUNCTION__);

	start_t = GetReferenceTime();
	phase = 33;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, b, x, &error);
	if ( error != 0 )
	{
			fprintf (stderr, "\nERROR during solution: %d", error);
			exit (3);
	}
	solve_t = GetReferenceTime() - start_t;
	
	fprintf(stderr, "\n%s: Solution time time %.6lf", __FUNCTION__, solve_t );

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

/*
 * Decomposes the matrix into L and U factors and then
 * uses them to solve multiple linear systems.
 */

void superlu_solve (const integer_t n, 
                    const integer_t nnz,
                    const integer_t nrhs,
                    integer_t *restrict colind,
                    integer_t *restrict rowptr,
                    complex_t *restrict aij,
                    complex_t *restrict x,
                    complex_t *restrict b )
{
	/* superlu overwrites the solution x on b vector */
	memcpy( x, b, n * nrhs * sizeof(complex_t));


	fprintf(stderr, "\n%s: line %d\n", __FUNCTION__, __LINE__ );
	/* local variables */
	spike_timer_t tstart_t, tend_t;

	tstart_t = GetReferenceTime();

  /* asub = colind and rowptr is rowptr */

    SuperMatrix              A, L, U;
    SuperMatrix              B, X;
    NRformat                *Astore;
    SCPformat               *Lstore;
    NCPformat               *Ustore;

    /*
     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = 0: natural ordering 
     *   permc_spec = 1: minimum degree ordering on structure of A'*A
     *   permc_spec = 2: minimum degree ordering on structure of A'+A
     *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
     */
     integer_t permc_spec = 3;

    /* row and column permutation vectors */
    integer_t *perm_c = (integer_t*) spike_malloc( ALIGN_INT, n, sizeof(integer_t)); /* column permutation vector */
    integer_t *perm_r = (integer_t*) spike_malloc( ALIGN_INT, n, sizeof(integer_t)); /* row permutations from partial pivoting */
    
    /* Extra variables */
    superlumt_options_t      superlumt_options;
    superlu_memusage_t       superlu_memusage;
    
    /* Default parameters to control factorization. */
    integer_t                nprocs = 1;
    fact_t                   fact   = EQUILIBRATE;
    trans_t                  trans  = TRANS;
    yes_no_t                 refact = NO;
    yes_no_t                 usepr  = NO;
    equed_t                  equed  = NOEQUIL;
    void                    *work;
    integer_t                info;
    integer_t                ldx = n;
    integer_t                lwork = 0;
    integer_t                panel_size = sp_ienv(1);
    integer_t                relax = sp_ienv(2);
    integer_t                i;
    real_t                  *R, *C;
    real_t                  *ferr, *berr;
    complex_t                u = 1.0;
    real_t                   drop_tol = 0.0;
    real_t                   rpg;
    real_t                   rcond;

    if ( lwork > 0 ) {
      work = SUPERLU_MALLOC(lwork);
      fprintf(stderr, "\nUse work space of size LWORK = " IFMT " bytes\n", lwork);
  
      if ( !work ) 
        SUPERLU_ABORT("SLINSOLX: cannot allocate work[]");
    }


  	fprintf(stderr, "Creating sparse matrix\n");

    dCreate_CompCol_Matrix(&A, n, n, nnz, aij, colind, rowptr, SLU_NR, SLU_D, SLU_GE);
    Astore = A.Store;
    fprintf(stderr, "Dimension " IFMT "x" IFMT "; # nonzeros " IFMT "\n", A.nrow, A.ncol, Astore->nnz);


    /* print A matrix */
    // complex_t *Avalues = (complex_t*) Astore->nzval;
    // integer_t *Acolind = (integer_t*) Astore->colind;
    // integer_t *Arowptr = (integer_t*) Astore->rowptr;

    /* create RHS matrix structures from input data */
    // dCreate_Dense_Matrix(&X, n, nrhs, b, n, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&B, n, nrhs, x, n, SLU_DN, SLU_D, SLU_GE);

    // dFillRHS(trans, nrhs, b, ldx, &A, &B);

    R    = (real_t*) spike_malloc( ALIGN_COMPLEX, A.nrow, sizeof(real_t)); 
    C    = (real_t*) spike_malloc( ALIGN_COMPLEX, A.ncol, sizeof(real_t));
    ferr = (real_t*) spike_malloc( ALIGN_COMPLEX, nrhs  , sizeof(real_t));
    berr = (real_t*) spike_malloc( ALIGN_COMPLEX, nrhs  , sizeof(real_t)); 

    /* case set up */
    get_perm_c(permc_spec, &A, perm_c);

    superlumt_options.nprocs            = nprocs;
    superlumt_options.fact              = fact;
    superlumt_options.trans             = trans;
    superlumt_options.refact            = refact;
    superlumt_options.panel_size        = panel_size;
    superlumt_options.relax             = relax;
    superlumt_options.usepr             = usepr;
    superlumt_options.drop_tol          = drop_tol;
    superlumt_options.diag_pivot_thresh = u;
    superlumt_options.SymmetricMode     = NO;
    superlumt_options.PrintStat         = YES;
    superlumt_options.perm_c            = perm_c;
    superlumt_options.perm_r            = perm_r;
    superlumt_options.work              = work;
    superlumt_options.lwork             = lwork;

    if ( !(superlumt_options.etree = intMalloc(n)) )
      SUPERLU_ABORT("Malloc fails for etree[].");
    if ( !(superlumt_options.colcnt_h = intMalloc(n)) )
      SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    if ( !(superlumt_options.part_super_h = intMalloc(n)) )
      SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    
    /* ------------------------------------------------------------
       SOLVE LINEAR SYSTEM
       ------------------------------------------------------------*/
     pdgssv(nprocs, &A, perm_c, perm_r, &L, &U, &B, &info);
    //pdgssvx(nprocs, &superlumt_options, &A, perm_c, perm_r,
    //   &equed, R, C, &L, &U, &B, &X, &rpg, &rcond,
    //   ferr, berr, &superlu_memusage, &info);
    printf("First system: psgssvx(): info " IFMT "\n----\n", info);


    if ( info == 0 || info == n+1 ) {
      printf("Recip. pivot growth       = %e\n", rpg);
      printf("Recip. condition number   = %e\n", rcond);
      printf("%8s%16s%16s\n", "rhs", "FERR", "BERR");
      
    	for (i = 0; i < nrhs; ++i)
    	  printf(IFMT "%16e%16e\n", i+1, ferr[i], berr[i]);
           
      Lstore = (SCPformat *) L.Store;
      Ustore = (NCPformat *) U.Store;
      
      printf("No of nonzeros in factor L = " IFMT "\n", Lstore->nnz);
      printf("No of nonzeros in factor U = " IFMT "\n", Ustore->nnz);
      printf("No of nonzeros in L+U      = " IFMT "\n", Lstore->nnz + Ustore->nnz - n);
      printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions " IFMT "\n",
      
      superlu_memusage.for_lu/1e6, superlu_memusage.total_needed/1e6,
      superlu_memusage.expansions);
         
      fflush(stdout);
    } else if ( info > 0 && lwork == -1 ) { 
      printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }

	tend_t = GetReferenceTime()  - tstart_t;


    /* print vectors */
    DNformat  *Bstore  = (DNformat* ) B.Store;
    complex_t *Bvalues = (complex_t*) Bstore->nzval;


    for(int i=0; i < 10; i++){
    	// (stderr, "x[%d] = %.lf b[%d] = %.lf\n", i, Xvalues[i], i, Bvalues[i]);
    	fprintf(stderr, "b[%d] = %f x[%d] %f\n", i, b[i], i, x[i]);
    }

    /* clean up and resume execution */
    //Destroy_CompCol_Matrix   (&A);
    //Destroy_SuperMatrix_Store(&B);
    //Destroy_SuperMatrix_Store(&X);


    spike_nullify (perm_r);
    spike_nullify (perm_c);
    spike_nullify (R);
    spike_nullify (C);
    spike_nullify (ferr);
    spike_nullify (berr);
    // spike_nullify (superlumt_options.etree);
    // spike_nullify (superlumt_options.colcnt_h);
    // spike_nullify (superlumt_options.part_super_h);

    if ( lwork == 0 ) {
        Destroy_SuperNode_SCP(&L);
        Destroy_CompCol_NCP(&U);
    } else if ( lwork > 0 ) {
        spike_nullify(work);
    }


	/* -------------------------------------------------------------------- */
	/* .. Compute Residual. */
	/* -------------------------------------------------------------------- */
    ComputeResidualOfLinearSystem( colind, rowptr, aij, x, b, n, nrhs );

    fprintf(stderr, "\nSuperLU solver took %.6lf seconds\n", tend_t);
};

void superlu_Factorize (const integer_t n, 
                    	const integer_t nnz,
                    	integer_t *restrict colind,
                    	integer_t *restrict rowptr,
                    	complex_t *restrict aij)
{
#ifdef jdafa
    NCformat *Astore;
    DNformat *Bstore;
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    SuperMatrix *AA; /* A in NC format used by the factorization routine.*/
    pdgstrf_options_t pdgstrf_options;
    Gstat_t  Gstat;
    
    yes_no_t refact = NO;
    yes_no_t usepr  = NO;
    trans_t  trans  = TRANS;
    int panel_size = sp_ienv(1);
    int relax      = sp_ienv(2);
    double diag_pivot_thresh = 1.;
    double drop_tol = 0.0;
    void *work      = NULL;
    int lwork       = 0;
    double   t; /* Temporary time */
    double   *utime;
    flops_t  *ops, flopcnt;

    /* ------------------------------------------------------------
       Allocate storage and initialize statistics variables. 
       ------------------------------------------------------------*/
    StatAlloc(n, nprocs, panel_size, relax, &Gstat);
    StatInit(n, nprocs, &Gstat);
    utime = Gstat.utime;
    ops   = Gstat.ops;

    /* ------------------------------------------------------------
       Initialize the option structure pdgstrf_options using the
       user-input parameters;
       Apply perm_c to the columns of original A to form AC.
       ------------------------------------------------------------*/
    pdgstrf_init(nprocs, refact, panel_size, relax,
		 diag_pivot_thresh, usepr, drop_tol, perm_c, perm_r,
		 work, lwork, AA, &AC, &pdgstrf_options, &Gstat);

    /* ------------------------------------------------------------
       Compute the LU factorization of A.
       The following routine will create nprocs threads.
       ------------------------------------------------------------*/
    pdgstrf(&pdgstrf_options, &AC, perm_r, L, U, &Gstat, info);

    flopcnt = 0;
    for (i = 0; i < nprocs; ++i) flopcnt += Gstat.procstat[i].fcops;
    ops[FACT] = flopcnt;

    /* ------------------------------------------------------------
       Solve the system A*X=B, overwriting B with X.
       ------------------------------------------------------------*/
    if ( *info == 0 ) {
        t = SuperLU_timer_();
		dgstrs (trans, L, U, perm_r, perm_c, B, &Gstat, info);
		utime[SOLVE] = SuperLU_timer_() - t;
		ops[SOLVE] = ops[TRISOLVE];
    }

    /* ------------------------------------------------------------
       Deallocate storage after factorization.
       ------------------------------------------------------------*/
    pdgstrf_finalize(&pdgstrf_options, &AC);
    if ( A->Stype == SLU_NR ) {
	Destroy_SuperMatrix_Store(AA);
	SUPERLU_FREE(AA);
    }

    /* ------------------------------------------------------------
       Print timings, then deallocate statistic variables.
       ------------------------------------------------------------*/
    PrintStat(&Gstat);
    StatFree(&Gstat);

    fprintf(stderr, "Factorizacion completada!\n");
#endif
};