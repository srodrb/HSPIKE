#include "spike_matrix.h"

matrix_t* matrix_LoadCSR(const char* filename)
{
	// local variables
	integer_t i;

	// TODO: create a proper rutine
	matrix_t* M = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));
	
	M->n      = 5;
	M->nnz    = 5;
	M->colind = (integer_t*) spike_malloc( ALIGN_INT    , M->nnz , sizeof(integer_t));
	M->rowptr = (integer_t*) spike_malloc( ALIGN_INT    , M->n +1, sizeof(integer_t));
	M->aij    = (complex_t*) spike_malloc( ALIGN_COMPLEX, M->nnz , sizeof(complex_t));

	//TODO: for now, it simply creates a diagonal matrix
	for(i=0; i<M->nnz; i++)
	{
		M->aij[i] = (complex_t) i;
	}

	for(i=0; i<M->nnz; i++)
	{
		M->colind[i] = (integer_t) i;
	}

	for(i=0; i<M->n +1; i++)
	{
		M->rowptr[i] = (integer_t) i;
	}

	matrix_Print(M, "\nTest matrix");

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

	fprintf(stderr, "\n\nMatrix coefficients\n");
	for(i=0; i<M->nnz; i++)
		fprintf(stderr, "%.3f ", M->aij[i]); 

	fprintf(stderr, "\n\nIndices\n");
	for(i=0; i<M->nnz; i++)
		fprintf(stderr, "%d ", M->colind[i]); 
	
	fprintf(stderr, "\n\nRow pointers\n");
	for(i=0; i<M->n +1; i++)
		fprintf(stderr, "%d ", M->rowptr[i]); 

	fprintf(stderr,"\n");

#endif
};

matrix_t* matrix_ExtractBlock ( matrix_t* M, 
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

	fprintf(stderr, "\nThe number of nnz elements in the block is %d, " \
					"number of rows in the block %d", nnz, B->n);
	
	// extract elements and correct indices
	nnz          = 0;
	rowind       = 1;
	B->rowptr[0] = 0;

	for(row=r0; row<rf; row++)
	{
		for(idx=M->colind[row]; idx<M->colind[row+1]; idx++)
		{
			col = M->colind[idx];

			fprintf(stderr,"\nRow value %d col value %d", row, col);

			if ((col >= c0) && (col < cf))
			{
				B->colind[nnz] = col - c0;
				B->aij[nnz]    = M->aij[idx];
				nnz++;
			}			
		}

		B->rowptr[rowind++] = nnz;
	}
	
	matrix_Print(B, "Matrix sub-block");

	return (B);
};
