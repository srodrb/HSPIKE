#include "spike_pardiso.h"

/* local functions */
static Error_t pardiso_Factorize(DirectSolverHander_t *handler );
static Error_t pardiso_ApplyFactorToRHS  ( DirectSolverHander_t *handler );
static Error_t pardiso_CleanUp( DirectSolverHander_t *handler );



DirectSolverHander_t *directSolver_CreateHandler(void)
{
	DirectSolverHander_t *handler = (DirectSolverHander_t*) spike_malloc( ALIGN_INT, 1, sizeof(DirectSolverHander_t));

	return (handler);
};

Error_t directSolver_Configure( DirectSolverHander_t *handler )
{
	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only       */
	/* necessary for the FIRST call of the PARDISO solver.                  */
	/* -------------------------------------------------------------------- */
	memset( (void*) handler->conf , 0, 64 * sizeof(MKL_INT));
	memset( (void*) handler->iparm, 0, 64 * sizeof(MKL_INT));

	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */
	handler->iparm[0]  =  1;    /* No solver default */
	handler->iparm[1]  =  2;    /* Fill-in reordering from METIS */
 	handler->iparm[2]  =  1;    /* Use all cores                 */
	handler->iparm[3]  =  0;    /* No iterative-direct algorithm */
	handler->iparm[4]  =  0;    /* No user fill-in reducing permutation */
	handler->iparm[5]  =  0;    /* Write solution into x */
	handler->iparm[6]  =  0;    /* Not in use */
	handler->iparm[7]  =  2;    /* Max numbers of iterative refinement steps */
	handler->iparm[8]  =  0;    /* Not in use */
	handler->iparm[9]  = 13;    /* Perturb the pivot elements with 1E-13 */
	handler->iparm[10] =  1;    /* Use nonsymmetric permutation and scaling MPS */
	handler->iparm[11] =  0;    /* Conjugate transposed/transpose solve */
	handler->iparm[12] =  1;    /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
	handler->iparm[13] =  0;    /* Output: Number of perturbed pivots */
	handler->iparm[14] =  0;    /* Not in use */
	handler->iparm[15] =  0;    /* Not in use */
	handler->iparm[16] =  0;    /* Not in use */
	handler->iparm[17] = -1;    /* Output: Number of nonzeros in the factor LU, -1 shows it */
	handler->iparm[18] = -1;    /* Output: Mflops for LU factorization, -1 shows it */
	handler->iparm[19] =  0;    /* Output: Numbers of CG Iterations */
	handler->iparm[34] =  1;    /* Pardiso use C-style indexing for ia and ja arrays */
	
	handler->maxfct    =  1;    /* Maximum number of numerical factorizations. */
	handler->mnum      =  1;    /* Which factorization to use. */
	handler->msglvl    =  0;    /* Print statistical information in file */
	handler->error     =  0;    /* Initialize error flag */

	/* -------------------------------------------------------------------- */
	/* .. Set the number of processors according to the OMP_NUM_THREADS   . */
	/* -------------------------------------------------------------------- */
    char *var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &handler->iparm[2] );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
    }

	/* Pardiso requieres this parameter at the factorization stage */
	/* but we still dont know anything about the rhs               */
	handler->nrhs   = 1;

    /* matrix type */
	handler->mtype = MTYPE_GEN_NOSYMM;

	/* statistical parameters */
	handler->ordering_t = 0.0;
	handler->factor_t   = 0.0;
	handler->solve_t    = 0.0;
	handler->clean_t    = 0.0;

	handler->rhs_block_count  = 0;
	handler->rhs_column_count = 0;

	return (SPIKE_SUCCESS);
};


Error_t directSolver_Factorize(DirectSolverHander_t *handler,
						const integer_t n,
						const integer_t nnz,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij)
{
	handler->colind = colind;
	handler->rowptr = rowptr;
	handler->aij    = aij;
	handler->n      = n;
	handler->nnz    = nnz;


	pardiso_Factorize( handler );
};


Error_t directSolver_SolveForRHS ( DirectSolverHander_t* handler,
                            const integer_t nrhs,
                            complex_t *restrict xij,
                            complex_t *restrict bij)
{
	/* update the value of rhs columns */
	handler->nrhs = nrhs;

	/* update statistics, keep track of RHS */
	handler->rhs_block_count  += 1;
	handler->rhs_column_count += nrhs;

	/* set the pointers to the values of the RHS */
	handler->xij = xij;
	handler->bij = bij;

	/* Forward and backward solution */
	pardiso_ApplyFactorToRHS( handler );

	return (SPIKE_SUCCESS);
};

Error_t directSolver_ShowStatistics( DirectSolverHander_t *handler )
{

#ifdef _ENABLE_TESTING_
    double avrhs; /* average time per rhs column solve */

    avrhs = handler->solve_t / (double) handler->rhs_column_count;

	fprintf(stderr, "\n\n--------------------------------------------------------");
	fprintf(stderr, "\n              BACKEND: PARDISO                         \n\n");
	fprintf(stderr, "\n Total number of RHS packs solved by this handler    : %d", handler->rhs_block_count      );
	fprintf(stderr, "\n Total number of RHS vectors solved by this hander   : %d", handler->rhs_column_count     );
	fprintf(stderr, "\n Number of OMP threads used                          : %d", handler->iparm[2]             );
	fprintf(stderr, "\n Reordering time                                     : %.6lf seconds", handler->ordering_t);
	fprintf(stderr, "\n Factorization time                                  : %.6lf seconds", handler->factor_t  );
	fprintf(stderr, "\n Solution time                                       : %.6lf seconds", handler->solve_t   );
	fprintf(stderr, "\n Release internal memory                             : %.6lf seconds", handler->clean_t   );	
	fprintf(stderr, "\n Av. time per RHS column                             : %.6lf seconds", avrhs);
	fprintf(stderr, "\n----------------------------------------------------------");
#endif

	return (SPIKE_SUCCESS);
};

Error_t directSolver_Finalize( DirectSolverHander_t *handler )
{
	pardiso_CleanUp(handler);

	spike_nullify(handler);

	return (SPIKE_SUCCESS);
};




static Error_t pardiso_ApplyFactorToRHS  ( DirectSolverHander_t *handler )
{
	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	MKL_INT phase;
	double ddum;          /* Double dummy */
	MKL_INT idum;         /* Integer dummy. */
	spike_timer_t start_t, end_t; /* local timer */

	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */

	/* -------------------------------------------------------------------- */
	/* .. Back substitution and iterative refinement. */
	/* -------------------------------------------------------------------- */
	//fprintf (stderr, "\n\n%s: Back substitution and iterative refinement...\n", __FUNCTION__);

	start_t = GetReferenceTime();
	phase = 33;
	PARDISO ( handler->conf,
		&handler->maxfct,
		&handler->mnum,
		&handler->mtype,
		&phase,
		&handler->n,
		handler->aij,
		handler->rowptr,
		handler->colind,
		&idum,
		&handler->nrhs,
		handler->iparm,
		&handler->msglvl,
		handler->bij,
		handler->xij,
		&handler->error);

	if ( handler->error != 0 )
	{
			fprintf (stderr, "\nERROR during solution: %d", handler->error);
			exit (3);
	}
	end_t = GetReferenceTime() - start_t;
	
	// fprintf(stderr, "\n%s: Solution time time %.6lf", __FUNCTION__, end_t );
	
	/* collect handler statistics */
	handler->solve_t += end_t;

	/* -------------------------------------------------------------------- */
	/* .. Compute Residual. */
	/* -------------------------------------------------------------------- */
    
    // ComputeResidualOfLinearSystem( colind, rowptr, aij, x, b, n, nrhs );

	/* -------------------------------------------------------------------- */
	/* .. Termination and release of memory. */
	/* -------------------------------------------------------------------- */

	return (SPIKE_SUCCESS);
};

static Error_t pardiso_Factorize(DirectSolverHander_t *handler )
{
	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	MKL_INT phase;
	double ddum;          						/* Double dummy */
	MKL_INT idum;         						/* Integer dummy. */
	spike_timer_t start_t, end_t;

	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */

	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only       */
	/* necessary for the FIRST call of the PARDISO solver.                  */
	/* -------------------------------------------------------------------- */

	/* -------------------------------------------------------------------- */
	/* .. Reordering and Symbolic Factorization. This step also allocates */
	/* all memory that is necessary for the factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 11;

	PARDISO ( handler->conf,
		&handler->maxfct,
		&handler->mnum,
		&handler->mtype,
		&phase,
		&handler->n,
		handler->aij,
		handler->rowptr,
		handler->colind,
		&idum,
		&handler->nrhs,
		handler->iparm,
		&handler->msglvl,
		&ddum,
		&ddum,
		&handler->error);

	if ( handler->error != 0 )
	{
			fprintf (stderr, "\nERROR during symbolic factorization: %d", handler->error);
			exit (1);
	}
	end_t = GetReferenceTime() - start_t;

	// fprintf(stderr, "\n%s: Reordering completed ... ", __FUNCTION__);
	// fprintf(stderr, "\n%s: Number of nonzeros in factors = %d", __FUNCTION__, handler->iparm[17]);
	// fprintf(stderr, "\n%s: Number of factorization MFLOPS = %d", __FUNCTION__, handler->iparm[18]);
	// fprintf(stderr, "\n%s: Reordering and Symbolic Factorization time %.6lf", __FUNCTION__, end_t );

	/* update handler statistics */
	handler->ordering_t += end_t;

	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	start_t = GetReferenceTime();
	phase = 22;
	PARDISO ( handler->conf,
		&handler->maxfct,
		&handler->mnum,
		&handler->mtype,
		&phase,
		&handler->n,
		handler->aij,
		handler->rowptr,
		handler->colind,
		&idum,
		&handler->nrhs,
		handler->iparm,
		&handler->msglvl,
		&ddum,
		&ddum,
		&handler->error);

	if ( handler->error != 0 )
	{
			fprintf (stderr, "\nERROR during numerical factorization: %d", handler->error);
			exit (2);
	}
	
	end_t = GetReferenceTime() - start_t;
	
	// fprintf(stderr, "\n%s: Factorization completed ... ", __FUNCTION__);
	// fprintf(stderr, "\n%s: Numerical factorization time %.6lf\n", __FUNCTION__, end_t );

	/* update handler statistics */
	handler->factor_t += end_t;


	return (SPIKE_SUCCESS);
};

static Error_t pardiso_CleanUp( DirectSolverHander_t *handler )
{

/* -------------------------------------------------------------------- */
/* .. Local variables. */
/* -------------------------------------------------------------------- */
	// MKL_INT mtype = MTYPE_GEN_NOSYMM;       /* Real unsymmetric matrix */
	// MKL_INT iparm[64]; /* Pardiso control parameters. */
	// MKL_INT conf[64];
	MKL_INT phase;
	double ddum;          /* Double dummy */
	MKL_INT idum;         /* Integer dummy. */
	spike_timer_t tstart_t;

/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
	tstart_t = GetReferenceTime();
	phase = -1;           /* Release internal memory. */
	PARDISO (   handler->conf,
				&handler->maxfct,
				&handler->mnum,
				&handler->mtype,
				&phase,
				&handler->n,
				&ddum,
				handler->rowptr,
				handler->colind,
				&idum,
				&handler->nrhs,
				handler->iparm,
				&handler->msglvl,
				&ddum,
				&ddum,
				&handler->error);

	handler->clean_t += GetReferenceTime() - tstart_t;

	return (SPIKE_SUCCESS);
};

/*
	Instead of using matrix_t structure, here we use the argument list that
	most back ends support.
 */
 Error_t directSolver_Solve (integer_t n,
 							integer_t nnz,
 							integer_t nrhs,
 							integer_t *restrict colind, // ja
							integer_t *restrict rowptr, // ia
							complex_t *restrict aij,
							complex_t *restrict x,
							complex_t *restrict b)
{
#ifdef _PARDISO_BACKEND_
	directSolver_Solve(n, nnz, nrhs, colind, rowptr, aij, x, b);
#else
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
	iparm[2]  =  1;    /* Use two cores */
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
	/* .. Set the number of processors according to the OMP_NUM_THREADS   . */
	/* -------------------------------------------------------------------- */
    char *var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &iparm[2] );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
    }

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
	phase = -1;           
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, &ddum, rowptr, colind, &idum, &nrhs,
					 iparm, &msglvl, &ddum, &ddum, &error);

	/* print some efficiency statistics */
	fprintf(stderr, "\n%s: Factor to solve ratio: %.6f\n", __FUNCTION__, factor_t / solve_t );

#endif

	return (SPIKE_SUCCESS);
};

