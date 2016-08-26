#include "spike_analysis_dm.h"

dm_schedule_t* spike_solve_analysis( matrix_t* A, const integer_t nrhs )
{
	/* local variables */
	integer_t i;
	integer_t nreg;    /* number of rows in a regular  block                     */
	integer_t nrem;    /* number of rows in a iregular block                     */
	uLong_t   HostMem; /* host available memory in bytes                         */
	integer_t p;       /* optimal number of matrix partitions in terms of memory */
	integer_t max_nrhs; /* maximum number of RHS on any sub-system                */

	/* ------------------ function body --------------------------------- */

	/* get host free memory resources */
	HostMem = get_maximum_av_host_memory();

	/* compute matrix bandwidth */
	matrix_ComputeBandwidth( A->n, A->colind, A->rowptr, A->aij, &A->ku, &A->kl );

	/* max_rhs is max(ku,kl,nrhs) */
	max_nrhs = ( A->ku   > A->kl ) ? A->ku   : A->kl;
	max_nrhs = ( max_nrhs > nrhs ) ? max_nrhs : nrhs;
	fprintf(stderr, "\n The maximum number of RHS on any sub-linear system is %d", max_nrhs);

	/* compute optimal number of partitions in terms of memory */
	p = compute_optimal_number_of_partitions( A, nrhs, HostMem );


	// gather information about hardware resources
	// TODO: get the number of cores properly
	// TODO: get the memory on the system properly
	// TODO: create a symbolic factorization routine
	// design a solve strategy



	nreg = (A->n / p);

	nrem = (A->n % p == 0) ? A->n/p : A->n - (A->n/p * (p-1));

	fprintf(stderr, "\nRegular block dimension "_I_", remainder "_I_, nreg, nrem);

	if ( nreg != nrem ) fprintf(stderr,"\nWarning: possible work unbalance");

	dm_schedule_t* S = (dm_schedule_t*) spike_malloc(ALIGN_INT, 1, sizeof(dm_schedule_t));
	S->max_n    		= ( nreg > nrem )   ? nreg : nrem;
	S->max_m    		= ( A->ku > A->kl ) ? A->ku : A->kl;
	S->p  	    		= p;
	S->max_nrhs         = max_nrhs;
	S->blockingDistance = 1; 
	S->n        		= (integer_t*) spike_malloc(ALIGN_INT, p +1, sizeof(integer_t));
	S->r        		= (integer_t*) spike_malloc(ALIGN_INT, p +1, sizeof(integer_t));
	S->ku       		= (integer_t*) spike_malloc(ALIGN_INT, p, sizeof(integer_t));
	S->kl       		= (integer_t*) spike_malloc(ALIGN_INT, p, sizeof(integer_t));

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

void schedule_Destroy( dm_schedule_t* S )
{
	spike_nullify( S->n  );
	spike_nullify( S->r  );
	spike_nullify( S->ku );
	spike_nullify( S->kl );

	spike_nullify( S );
};

void schedule_Print (dm_schedule_t* S)
{
	fprintf(stderr,"\nNumber of diagonal blocks:" _I_, S->p);

	for(integer_t i=0; i<S->p; i++)
		fprintf(stderr,"\n\t"_I_"-th block goes from " _I_"-th to " _I_"-th row. ku "_I_" kl "_I_,  i+1, S->n[i], S->n[i+1], S->ku[i], S->kl[i]);

	fprintf(stderr, "\nReduced system dimensions:");

	for(integer_t i=0; i<S->p; i++)
		fprintf(stderr,"\n\t"_I_"-th block goes from "_I_"-th to "_I_"-th row. ku "_I_" kl "_I_, i+1, S->r[i], S->r[i+1], S->ku[i], S->kl[i]);

	fprintf(stderr, "\nReduced system dimensions:");

	fprintf(stderr,"\n\nMax nrhs %d, blocking distance %d\n\n\n", S->max_nrhs, S->blockingDistance );
};

/*
	This function computes the max. free memory of the host no in bytes.
	This number is used to compute the optimal solving strategy.
 */

uLong_t get_maximum_av_host_memory( void )
{
	 struct sysinfo si;
	 sysinfo (&si);

	 printf ("\tHost total RAM   : %5.2f GB\n", bytesToGb( si.totalram));
	 printf ("\tHost free RAM    : %5.2f GB\n", bytesToGb( si.freeram ));
	
	return ((uLong_t) si.freeram );
};

/*
	Computes the memory requirements for each number of partitions.

	For now, I'll assume that ku and kl are equal and constant for the entire matrix.
*/
integer_t compute_optimal_number_of_partitions( matrix_t *A, integer_t nrhs, uLong_t HostMem )
{
	/* local variables */
	integer_t i;           /* dummy integer                          */
	integer_t p;           /* optimal number of partitions           */
	integer_t p_max;       /* maximum theoretical number of parts.   */
	uLong_t A_cost;        /* weigth in bytes of the entire A matrix */
	uLong_t f_cost;
	uLong_t x_cost;

	integer_t ni; /* number of rows on a Aij sub-matrix */

	/* compute base costs */
	A_cost = A->nnz * sizeof(complex_t) + A->n * sizeof(complex_t) + (A->n + 1) * sizeof(complex_t);
	x_cost = A->n   * sizeof(complex_t);
	f_cost = A->n   * sizeof(complex_t);

	/* compute maximum number of partitions                                   */
	/* Instead of dividing by A->ku + A->kl y doble this number, this ensures */
	/* a factor of 2 of the reduction ratio, at least                         */
	p_max = floor( ((double) A->n) / ((double) ( A->ku + A->kl + A->ku + A->kl )) );

	fprintf(stderr, "\n The maximum number of partitions is %d", p_max);

	/* estimate the costs for different values of p */
	for(i=0; i < p_max; i++){
		ni = 3;
	}

	integer_t size;
	MPI_Comm_size ( MPI_COMM_WORLD, &size);
	if( MASTER_WORKING == 0) return (size - 1);
	else return (size);
};


