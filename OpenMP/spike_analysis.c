#include "spike_analysis.h"

sm_schedule_t* spike_solve_analysis( matrix_t* A, const integer_t nrhs, const integer_t p)
{
	if ( p > A->n ){
		fprintf(stderr, "Number of partitions is too high. Unable to create the partition table.\n");
		abort();
	}


	// local variables
	integer_t i;
	integer_t nreg;  // regular block dimension
	integer_t nrem;  // irregular block dimension

	// gather information about hardware resources
	// TODO: get the number of cores properly
	// TODO: get the memory on the system properly
	// TODO: create a symbolic factorization routine
	// design a solve strategy

	matrix_ComputeBandwidth( A->n, A->colind, A->rowptr, A->aij, &A->ku, &A->kl );

	nreg = (A->n / p);

	nrem = (A->n % p == 0) ? A->n/p : A->n - (A->n/p * (p-1));

	fprintf(stderr, "\nRegular block dimension "_I_", remainder "_I_, nreg, nrem);

	if ( nreg != nrem )
	{
		fprintf(stderr,"\nWarning: possible work unbalance");
	}

	sm_schedule_t* S = (sm_schedule_t*) spike_malloc(ALIGN_INT, 1, sizeof(sm_schedule_t));
	S->max_n = ( nreg > nrem )   ? nreg : nrem;
	S->max_m = ( A->ku > A->kl ) ? A->ku : A->kl;
	S->p  = p;
	S->n     = (integer_t*) spike_malloc(ALIGN_INT, p +1, sizeof(integer_t));
	S->r     = (integer_t*) spike_malloc(ALIGN_INT, p +1, sizeof(integer_t));
	S->ku    = (integer_t*) spike_malloc(ALIGN_INT, p, sizeof(integer_t));
	S->kl    = (integer_t*) spike_malloc(ALIGN_INT, p, sizeof(integer_t));

	memset(S->n, 0, (p +1) * sizeof(integer_t));
	memset(S->r, 0, (p +1) * sizeof(integer_t));

	for(integer_t i=0; i < S->p; i++)
	{
		S->ku[i] = A->ku;
		S->kl[i] = A->kl;
	}

	for(i=1; i < S->p; i++)
	{
		/* original linear system starting and endind rows for each partition */
		S->n[i] = (integer_t) i*nreg;
		/* reduced linear system starting and endind rows for each partition */
		S->r[i] = S->r[i-1] + S->ku[i] + S->kl[i];
	}

	// remainder element
	S->n[S->p] = (integer_t) A->n;
	S->r[S->p] = S->r[S->p-1] + S->ku[S->p-1] + S->kl[S->p -1];

	/* print schedule information */
	schedule_Print(S);

	return (S);
};

void schedule_Destroy( sm_schedule_t* S )
{
	spike_nullify( S->n  );
	spike_nullify( S->r  );
	spike_nullify( S->ku );
	spike_nullify( S->kl );

	spike_nullify( S );
};

void schedule_Print (sm_schedule_t* S)
{
	fprintf(stderr,"\nNumber of diagonal blocks:" _I_, S->p);

	for(integer_t i=0; i<S->p; i++)
		fprintf(stderr,"\n\t"_I_"-th block goes from " _I_"-th to " _I_"-th row. ku "_I_" kl "_I_,  i+1, S->n[i], S->n[i+1], S->ku[i], S->kl[i]);

	fprintf(stderr, "\nReduced system dimensions:");

	for(integer_t i=0; i<S->p; i++)
		fprintf(stderr,"\n\t"_I_"-th block goes from "_I_"-th to "_I_"-th row. ku "_I_" kl "_I_, i+1, S->r[i], S->r[i+1], S->ku[i], S->kl[i]);


	fprintf(stderr,"\n\n");
};
