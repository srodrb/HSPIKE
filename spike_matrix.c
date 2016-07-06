#include "spike_matrix.h"
#include "spike_common.h"

matrix_t* matrix_LoadCSR(const char* filename)
{
	// local variables
	integer_t dtype;

	matrix_t* M = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));

	// Open File
	FILE* f = spike_fopen( filename, "rb");

	// read number of rows
	spike_fread( &M->n, sizeof(integer_t), 1, f );

	// read number of nnz
	spike_fread( &M->nnz, sizeof(integer_t), 1, f );

	// read data type, just to check everything is fine
	spike_fread( &dtype, sizeof(integer_t), 1, f );

	// allocate space for matrix coefficients and load them
	M->aij    = (complex_t*) spike_malloc( ALIGN_COMPLEX, M->nnz , sizeof(complex_t));
	spike_fread( (void*) M->aij, sizeof(complex_t), M->nnz, f );

	// allocate space for matrix indices and load them
	M->colind = (integer_t*) spike_malloc( ALIGN_INT    , M->nnz , sizeof(integer_t));
	spike_fread( (void*) M->colind, sizeof(integer_t), M->nnz, f );

	// allocate space for matrix row pointers and load them
	M->rowptr = (integer_t*) spike_malloc( ALIGN_INT    , M->n +1, sizeof(integer_t));
	spike_fread( (void*) M->rowptr, sizeof(integer_t), M->n + 1, f );

	// close file
	spike_fclose(f);


	return (M);
};

void matrix_Deallocate (matrix_t* M)
{
	spike_nullify ( M->colind );
	spike_nullify ( M->rowptr );
	spike_nullify ( M->aij    );
	spike_nullify ( M );
};

void matrix_Print(matrix_t* M, const char* msg)
{
#ifdef _ENABLE_TESTING_
	integer_t i;

	fprintf(stderr, "\n%s: %s", __FUNCTION__, msg);

	fprintf(stderr, "\n\n\tMatrix dimension: %d, nnz: %d\n", M->n, M->nnz);

	fprintf(stderr, "\n\n\tMatrix coefficients\n");
	for(i=0; i<M->nnz; i++)
		fprintf(stderr, "\t%.3f ", M->aij[i]);

	fprintf(stderr, "\n\n\tIndices\n");
	for(i=0; i<M->nnz; i++)
		fprintf(stderr, "\t%d ", M->colind[i]);

	fprintf(stderr, "\n\n\tRow pointers\n");
	for(i=0; i<M->n +1; i++)
		fprintf(stderr, "\t%d ", M->rowptr[i]);

	fprintf(stderr,"\n");

#endif
};

matrix_t* matrix_Extract (  matrix_t* M,
														const integer_t r0,
														const integer_t rf,
														const integer_t c0,
														const integer_t cf)
{
	// TODO: integrar el calculo del BW en la extraccion del bloque
	// TODO: hacer una version single-pass.

	/*
	 * Extracts a sub-block of the original matrix and returns
	 * it as a new sparse matrix.
	 * It does perform a single-pass copy, namely, it approximates
	 * the number of nnz elements in the block.
	 * It is also needed to correct the values of the indices.
	 */

	// local variables
	integer_t nnz;
	integer_t rowind;
	integer_t nrows;
 	integer_t	idx;
 	integer_t	row;
 	integer_t	col;

	// count the number of nnz inside the block
	nnz = 0;
	for(row=r0; row<rf; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];

			if ((col >= c0) && (col < cf))
				nnz++;
		}
	}

	// allocate matrix space
	matrix_t* B = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));

	B->n       = rf - r0;
	B->nnz     = nnz;
	B->colind  = (integer_t*) spike_malloc( ALIGN_INT    , nnz     , sizeof(integer_t));
	B->rowptr  = (integer_t*) spike_malloc( ALIGN_INT    , B->n + 1, sizeof(integer_t));
	B->aij     = (complex_t*) spike_malloc( ALIGN_COMPLEX, nnz     , sizeof(complex_t));

	// extract elements and correct indices
	nnz          = 0;
	rowind       = 1;
	B->rowptr[0] = 0;

	for(row=r0; row<rf; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];

			if ((col >= c0) && (col < cf))
			{
				B->colind[nnz] = col - c0;
				B->aij[nnz]    = M->aij[idx];
				nnz++;
			}
		}

		B->rowptr[rowind++] = nnz;
	}

	return (B);
};

/*
 * returns 1 if both matrices are equal, 0 otherwise
 */
Error_t matrix_AreEqual( matrix_t* A, matrix_t* B )
{
	if( A->n != B->n )
	{
		fprintf(stderr, "Dimension mismatch\n");
		return (0);
	}

	if( A->nnz != B->nnz )
	{
		fprintf(stderr, "Number of nnz is not the same\n");
		return (0);
	}

	for (integer_t i = 0; i < A->nnz; i++)
	{
		if( A->aij[i] != B->aij[i] )
		{
			fprintf(stderr, "%d-th coefficents are not equal\n", i);
			return (0);
		}
	}

	for (integer_t i = 0; i < A->nnz; i++)
	{
		if( A->colind[i] != B->colind[i] )
		{
			fprintf(stderr, "%d-th indices are not equal\n", i);
			return (0);
		}
	}

	for (integer_t i = 0; i < A->n+1; i++)
	{
		if( A->rowptr[i] != B->rowptr[i] )
		{
			fprintf(stderr, "%d-th row pointers are not equal\n", i);
			return (0);
		}
	}

	return (1);
};

block_t* block_Extract (  matrix_t* M,
													const integer_t r0,
													const integer_t rf,
													const integer_t c0,
													const integer_t cf)
{
	integer_t row, col, idx;

	// allocates the -dense- block
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));

	B->n   = rf - r0;
	B->m   = cf - c0;
	B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, B->n * B->m, sizeof(complex_t));

	// extract the elements, correct the indices and insert them into the dense block
	for(row=r0; row<rf; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];

			if ((col >= c0) && (col < cf))
				B->aij[ (row -r0) * B->m + (col - c0)] = M->aij[idx];
		}
	}

	return (B);
};

/* deallocates a block structure */
void block_Deallocate(block_t* B)
{
	spike_nullify( B->aij );
	spike_nullify( B );
};

void block_Print( block_t* B, const char* msg )
{
	integer_t row, col;

	fprintf(stderr, "\n%s\n", msg);

	for (row = 0; row < B->n; row++)
	{
		fprintf(stderr, "\n\t");

		for (col = 0; col < B->m; col++)
			fprintf(stderr, "%f   ", B->aij[row * B->m + col]);

	}
	fprintf(stderr, "\n");
};

/*
 * returns 1 if both block are equal, 0 otherwise
 */
Error_t block_AreEqual( block_t* A, block_t* B )
{
	if( A->n != B->n || A->m != B->m)
	{
		fprintf(stderr, "Dimension mismatch\n");
		return (0);
	}

	for (integer_t i = 0; i < A->n * A->m; i++)
	{
		if( A->aij[i] != B->aij[i] )
		{
			fprintf(stderr, "%d-th coefficents are not equal\n", i);
			return (0);
		}
	}

	return (1);
};

/*
	Creates an empty block of dimension n,m.
	It is intended to create buffers for system_solve call.
*/
block_t* block_Empty( const integer_t m, const integer_t n)
{
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->n = n;
	B->m = m;
	B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, m * n, sizeof(complex_t));

	return (B);
}
