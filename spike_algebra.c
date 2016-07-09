#include "spike_algebra.h"

void reorder_metis( matrix_t* A, integer_t* colperm )
{

};

void reorder_fieldler( matrix_t* A, integer_t* colperm, integer_t* scale )
{

};

void reorder_rcm ( matrix_t* A, integer_t* colperm )
{

};

/*
	Instead of using matrix_t structure, here we use the argument list that
	most back ends support.
 */
Error_t system_solve ( integer_t *colind, // ja
										   integer_t *rowptr, // ia
										   complex_t *aij,
										   complex_t *x,
										   complex_t *b,
										   integer_t  n,
										   integer_t  nrhs)
{
	if ( x == NULL ){
		fprintf(stderr, "INFO: x vector is not supplied, solution will be stored "
										" on b vector\n");
	}

	MKL_INT mtype = 11;       /* Real unsymmetric matrix */

	/* RHS and solution vectors. */
	double res, res0;
	void *pt[64];
	/* Pardiso control parameters. */
	MKL_INT iparm[64];
	MKL_INT maxfct, mnum, phase, error, msglvl;
	/* Auxiliary variables. */
	double ddum;          /* Double dummy */
	MKL_INT idum;         /* Integer dummy. */
	char *uplo;

/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
	for(MKL_INT i=0; i<64; i++) iparm[i] = 0;

	iparm[0] = 1;         /* No solver default */
	iparm[1] = 2;         /* Fill-in reordering from METIS */
	iparm[3] = 0;         /* No iterative-direct algorithm */
	iparm[4] = 0;         /* No user fill-in reducing permutation */
	iparm[5] = 0;         /* Write solution into x */
	iparm[6] = 0;         /* Not in use */
	iparm[7] = 2;         /* Max numbers of iterative refinement steps */
	iparm[8] = 0;         /* Not in use */
	iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
	iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
	iparm[11] = 0;        /* Conjugate transposed/transpose solve */
	iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
	iparm[13] = 0;        /* Output: Number of perturbed pivots */
	iparm[14] = 0;        /* Not in use */
	iparm[15] = 0;        /* Not in use */
	iparm[16] = 0;        /* Not in use */
	iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU, -1 shows it */
	iparm[18] = -1;       /* Output: Mflops for LU factorization, -1 shows it */
	iparm[19] = 0;        /* Output: Numbers of CG Iterations */
	iparm[34] = 1;        /* Pardiso use C-style indexing for ia and ja arrays */
	maxfct = 1;           /* Maximum number of numerical factorizations. */
	mnum = 1;         /* Which factorization to use. */
	msglvl = 0;           /* Print statistical information in file */
	error = 0;            /* Initialize error flag */
/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* -------------------------------------------------------------------- */
	for (MKL_INT i = 0; i < 64; i++ ) { pt[i] = 0; }
/* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates */
/* all memory that is necessary for the factorization. */
/* -------------------------------------------------------------------- */
	phase = 11;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if ( error != 0 )
	{
			printf ("\nERROR during symbolic factorization: %d", error);
			exit (1);
	}

	printf ("\nReordering completed ... ");
	printf ("\nNumber of nonzeros in factors = %d", iparm[17]);
	printf ("\nNumber of factorization MFLOPS = %d", iparm[18]);
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */
	phase = 22;
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	if ( error != 0 )
	{
			printf ("\nERROR during numerical factorization: %d", error);
			exit (2);
	}
	printf ("\nFactorization completed ... ");
/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
	phase = 33;
	iparm[11] = 0;
	uplo = "non-transposed";

	printf ("\n\nSolving %s system...\n", uplo);
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, aij, rowptr, colind, &idum, &nrhs, iparm, &msglvl, b, x, &error);
	if ( error != 0 )
	{
			printf ("\nERROR during solution: %d", error);
			exit (3);
	}

// Compute residual

// Check residual


/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
	phase = -1;           /* Release internal memory. */
	PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
					 &n, &ddum, rowptr, colind, &idum, &nrhs,
					 iparm, &msglvl, &ddum, &ddum, &error);

	// solve the system

	// check for residual
	return (SPIKE_SUCCESS);
};

void symbolic_factorization ( matrix_t* A )
{

};

void compute_bandwidth( matrix_t* A )
{
	// TODO: es posible calcular el bw mas rapidamente accediendo
	// solamente a las posiciones extremas de las filas.


	integer_t row, col, idx;
	integer_t ku = 0;
	integer_t kl = 0;

	for(row = 0; row < A->n; row++)
	{
		for(idx = A->rowptr[row]; idx < A->rowptr[row+1]; idx++)
		{
			col = A->colind[idx];

			ku = ((row - col) < ku) ? (row - col) : ku;
			kl = ((col - row) < kl) ? (col - row) : kl;
		}
	}

	A->ku = abs(ku);
	A->kl = abs(kl);

	fprintf(stderr,"\nBandwitdh computed: (upper,lower) (%d,%d)", A->ku, A->kl);
};
