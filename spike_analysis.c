#include "spike_analysis.h"

sm_schedule_t* spike_solve_analysis( matrix_t* A, const integer_t nrhs )
{
	// local variables
	integer_t i;
	integer_t p = 2; // number of partitions
	integer_t nreg;  // regular block dimension
	integer_t nrem;  // irregular block dimension

	// gather information about hardware resources
	// TODO: get the number of cores properly
	// TODO: get the memory on the system properly
	// TODO: create a symbolic factorization routine
	// design a solve strategy

	compute_bandwidth( A );

	nreg = (A->n / p);
	nrem = (A->n % p == 0) ? A->n/p : A->n - (A->n/p * (p-1));

	fprintf(stderr, "\nRegular block dimension %d, remainder %d", nreg, nrem);

	if ( nreg != nrem )
	{
		fprintf(stderr,"\nWarning: possible work unbalance");
	}

	sm_schedule_t* S = (sm_schedule_t*) spike_malloc(ALIGN_INT, 1, sizeof(sm_schedule_t));
	S->p  = p;
	S->n  = (integer_t*) spike_malloc(ALIGN_INT, p +1, sizeof(integer_t));
	S->ku = (integer_t*) spike_malloc(ALIGN_INT, p, sizeof(integer_t));
	S->kl = (integer_t*) spike_malloc(ALIGN_INT, p, sizeof(integer_t));

	S->n[0] = (integer_t) 0;

	for(i=1; i < S->p; i++)
	{
		S->n[i] = (integer_t) i*nreg;
	}
	// remainder element
	S->n[S->p] = (integer_t) A->n;

	for(integer_t i=0; i < S->p; i++)
	{
		S->ku[i] = A->ku;
		S->kl[i] = A->kl;
	}

	S->kl[0]       = (integer_t) 0;
	S->ku[S->p -1] = (integer_t) 0;

	schedule_Print(S);

	return (S);
};

void schedule_Destroy( sm_schedule_t* S )
{
	spike_nullify( S->n  );
	spike_nullify( S->ku );
	spike_nullify( S->kl );

	spike_nullify(S);
};

void schedule_Print (sm_schedule_t* S)
{
	// function body
	fprintf(stderr,"\nNumber of diagonal blocks: %d", S->p);

	for(integer_t i=0; i<S->p; i++)
	{
		fprintf(stderr,"\n\t%d-th block goes from %d-th to %d-th row. ku %d kl %d", i+1, S->n[i], S->n[i+1], S->ku[i], S->kl[i]);
	}

	fprintf(stderr,"\n\n");
};
