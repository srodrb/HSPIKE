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


void system_solve( matrix_t* A, complex_t* x, complex_t* b, const integer_t nrhs)
{
	// solve the system
	
	// check for residual	
};

void symbolic_factorization ( matrix_t* A )
{

};

void compute_bandwidth( matrix_t* A )
{
	//TODO: compute the bandwidth properly
	A->ku = 1;
	A->kl = 1;
	
	fprintf(stderr,"\nBandwitdh computed: (upper,lower) (%d,%d)", A->ku, A->kl);
};
