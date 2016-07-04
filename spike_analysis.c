#include "spike_analysis.h"

sm_schedule_t* spike_solve_analysis( matrix_t* A, const integer_t nrhs )
{
	// local variables
	integer_t i;
	integer_t p = 1; // number of partitions
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
	S->p = p;
	S->interval = (interval_t*) spike_malloc(ALIGN_INT, p, sizeof(interval_t));

	for(i=0; i<S->p -1; i++)
	{
		S->interval[i] = (interval_t) {i*nreg, (i+1)*nreg};
	}

	S->interval[S->p -1] = (interval_t) {(S->p -1)*nreg, A->n};
	

	schedule_Print(S);

	return (S);
};

void schedule_Destroy( sm_schedule_t* S )
{
	spike_nullify(S->interval);
	spike_nullify(S);
};

void schedule_Print (sm_schedule_t* S)
{
	// local variables
	integer_t i;

	// function body
	fprintf(stderr,"\nNumber of diagonal blocks: %d", S->p);
	
	for(i=0; i<S->p; i++)
	{
		fprintf(stderr,"\n\t%d-th block goes from %d-th to %d-th row", \
				i+1, S->interval[i].r0, S->interval[i].rf);
	}

	fprintf(stderr,"\n\n");
};

