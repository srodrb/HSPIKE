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
Error_t system_solve ( integer_t* colind,
										integer_t* rowptr,
										complex_t* aij,
										complex_t* x,
										complex_t* b,
										const integer_t n,
										const integer_t nrhs)
{
	if ( x == NULL ){
		fprintf(stderr, "INFO: x vector is not supplied, solution will be stored\
										 on b vector\n");
	}


	// solve the system

	// check for residual
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
