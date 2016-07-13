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

/*
	Creates an empty CSR sparse matriz of dimension n and nnz elements
*/
static matrix_t* matrix_CreateEmpty( const integer_t n, const integer_t nnz )
{
	matrix_t* R = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));
	R->n        = n;
	R->nnz      = nnz;
	R->ku       = 0;
	R->kl       = 0;
	R->K        = 0;
	R->colind   = (integer_t*) spike_malloc( ALIGN_INT    , R->nnz, sizeof(integer_t));
	R->rowptr   = (integer_t*) spike_malloc( ALIGN_INT    , R->n+1, sizeof(integer_t));
	R->aij      = (complex_t*) spike_malloc( ALIGN_COMPLEX, R->nnz, sizeof(complex_t));

	memset( (void*) R->colind, 0, (R->nnz) * sizeof(integer_t));
	memset( (void*) R->rowptr, 0, (R->n+1) * sizeof(integer_t));
	memset( (void*) R->aij   , 0, (R->nnz) * sizeof(complex_t));

	return (R);
}

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

Error_t matrix_PrintAsDense( matrix_t* A, const char* msg)
{
  const integer_t nrows = A->n;
  const integer_t ncols = A->n;
	complex_t value;

	if    (msg) { fprintf(stderr, "\n%s: %s\n\n", __FUNCTION__, msg);}
	else        { fprintf(stderr, "\n%s\n\n"    , __FUNCTION__ ); }

  complex_t *D = (complex_t*) spike_malloc( ALIGN_COMPLEX, ncols * nrows, sizeof(complex_t));

  memset( (void*) D, 0, nrows * ncols * sizeof(complex_t));

  for(integer_t row = 0; row < nrows; row++){
    for(integer_t idx = A->rowptr[row]; idx < A->rowptr[row+1]; idx++ ){
      integer_t col = A->colind[idx];
      D[ row * ncols + col] = A->aij[idx];
    }
  }

  if (msg) fprintf(stderr, "\n%s: %s\n\n", __FUNCTION__, msg);

  for(integer_t row = 0; row < nrows; row++){
    for(integer_t col = 0; col < ncols; col++){
			value = D[row * ncols + col];

			if ( number_IsLessThan ( value, __zero ) == True )
				fprintf(stderr, "%.5f  ", value);
			else
				fprintf(stderr, " %.5f  ", value);

    }
    fprintf(stderr, "\n");
  }

  spike_nullify(D);

  return (SPIKE_SUCCESS);
};

matrix_t* matrix_ExtractMatrix (  matrix_t* M,
														const integer_t r0,
														const integer_t rf,
														const integer_t c0,
														const integer_t cf)
{
	// complex_t* restrict aij    __attribute__ ((aligned (ALIGN_COMPLEX))) = M->aij;
	// integer_t* restrict coling __attribute__ ((aligned (ALIGN_INT    ))) = M->colind;
	// integer_t* restrict rowptr __attribute__ ((aligned (ALIGN_INT    ))) = M->rowptr;


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
	matrix_t* B = matrix_CreateEmpty( rf - r0, nnz );

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
		if( number_IsEqual(A->aij[i],B->aij[i]) == False )
		{
			fprintf(stderr, "%d-th coefficents are not equal\n", i);
			return (0);
		}
	}

	for (integer_t i = 0; i < A->nnz; i++)
	{
		if( A->colind[i] != B->colind[i]  )
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

block_t* matrix_ExtractBlock (  matrix_t* M,
													const integer_t r0,
													const integer_t rf,
													const integer_t c0,
													const integer_t cf,
													blocktype_t type )
{
	integer_t row, col, idx;

	// allocates the -dense- block
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->type = type;
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
	complex_t value;

	fprintf(stderr, "\n%s\n", msg);

	for (row = 0; row < B->n; row++)
	{
		fprintf(stderr, "\n\t");
		for (col = 0; col < B->m; col++) {
			value = B->aij[row * B->m + col];

			if ( number_IsLessThan( value, __zero ) == True )
				fprintf(stderr, "%.5f  ", value);
			else
				fprintf(stderr, " %.5f  ", value);
		}
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
		if( number_IsEqual( A->aij[i], B->aij[i]) == False )
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
block_t* block_Empty( const integer_t n, const integer_t m, blocktype_t type)
{
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->type = type;
	B->n = n;
	B->m = m;
	B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, m * n, sizeof(complex_t));

	memset( (void*) B->aij, 0, m * n * sizeof(complex_t));

	return (B);
};

Error_t block_InitializeToValue( block_t* B, const complex_t value )
{
	complex_t *restrict aij = B->aij;

	for(integer_t i=0; i < (B->n * B->m); i++)
		aij[i] = value;

	return (SPIKE_SUCCESS);
};

/*
	Computes an array, similar to R->n that gives the number of rows per block.
 */
static integer_t* ComputeReducedSytemDimensions( integer_t partitions, integer_t *ku, integer_t *kl)
{
	integer_t* nr = (integer_t*) spike_malloc( ALIGN_INT, partitions + 1, sizeof(integer_t));
	memset( (void*) nr, 0, (partitions +1) * sizeof(integer_t));

	for(integer_t i=0; i < partitions; i++) {
		nr[i+1] = nr[i] + (ku[i] + kl[i]);
	}

	return (nr);
}

/*
  p = numero de particiones que lo originan
  k = (array) contiene el numero de columnas de cada bloque
  n = (array) contiene el numero de filas de cada bloque
 */
matrix_t* matrix_CreateEmptyReduced( const integer_t TotalPartitions, integer_t *n, integer_t *ku, integer_t *kl )
{
	// local variables
	integer_t nnz, rows;

	// compute matrix dimensions and allocate the structure
	GetNnzAndRowsUpToPartition(TotalPartitions, TotalPartitions, ku, kl, &nnz, &rows );
	matrix_t* R = matrix_CreateEmpty( rows, nnz );

	// reduced system dimensions
	integer_t* nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);

  // initialize blocks
  for(integer_t p=0; p < TotalPartitions; p++)
	{
		GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

		/* ------------- top spike elements ---------------- */
#ifdef _DEBUG_MATRIX_
		fprintf(stderr, "\tPartition %d, top part covers from %d to %d\n", p, nr[p], nr[p] + ku[p] );
#endif

		for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
			if ( p > 0 )// add Wi elements
			for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) {
				// R->aij[nnz] = 3.3;
				R->colind[nnz++] = col;
			}

			// add diagonal element
			R->colind[nnz  ] = row;
			R->aij   [nnz++] = (complex_t) __unit;

			// add Wi elements
			if ( p < (TotalPartitions - 1)) // add Vi elements
			for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) {
				// R->aij[nnz] = 2.2;
				R->colind[nnz++] = col;
			}

			if ( p == 0 )
				R->rowptr[row+1] = R->rowptr[row] + (ku[p] +1);
			else if ( p == (TotalPartitions -1))
				R->rowptr[row+1] = R->rowptr[row] + (kl[p] +1);
			else
				R->rowptr[row+1] = R->rowptr[row] + (ku[p] + kl[p] +1);
		}

		/* ------------- Bottom spike elements ---------------- */
#ifdef _DEBUG_MATRIX_
		fprintf(stderr, "\tPartition %d, bottom part covers from %d to %d\n", p, nr[p] +ku[p], nr[p+1] );
#endif

		for(integer_t row = (nr[p] + ku[p]); row < nr[p+1]; row++ ) {
			if ( p > 0 )// add Wi elements
			for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) {
				// R->aij[nnz] = 4.4;
				R->colind[nnz++] = col;
			}

			// add diagonal element
			R->colind[nnz  ] = row;
			R->aij   [nnz++] = (complex_t) __unit;

			// add Wi elements
			if ( p < (TotalPartitions - 1)) // add Vi elements
			for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) {
				// R->aij[nnz] = 5.5;
				R->colind[nnz++] = col;
			}

			if ( p == 0 )
				R->rowptr[row+1] = R->rowptr[row] + (ku[p] +1);
			else if ( p == (TotalPartitions -1))
				R->rowptr[row+1] = R->rowptr[row] + (kl[p] +1);
			else
				R->rowptr[row+1] = R->rowptr[row] + (ku[p] + kl[p] +1);
		}
	}

	// clean up
	spike_nullify( nr );

  return (R);
}

/*
  Computes the number of nnz in the reduced system.

  p = total number of partitions
  n = number of rows per block (array)
  ku = upper bandwidth for Vi blocks
  kl = lower bandwidth for Wi blocks
  nnz = number of nnz elements
  dim = total number of rows in the reduced system
*/
Error_t GetNnzAndRowsUpToPartition ( const integer_t TotalPartitions, const integer_t CurrentPartition, integer_t* ku, integer_t *kl, integer_t *nnz, integer_t *FirstBlockRow )
{
	/* Compute the number of nnz elements up to the actual partition */
  *nnz      = 0;
  for(integer_t p = 0; p < CurrentPartition; p++)
  {
		if ( p == 0 ){
			*nnz += ku[p] * (ku[p] + kl[p]);
		}
		else if ( p == (TotalPartitions -1)) {
			*nnz += kl[p] * (ku[p] + kl[p]);
		}
		else{
			*nnz += kl[p] * (ku[p] + kl[p]) + ku[p] * (ku[p] + kl[p]);
		}

		// add diagonal elements
		*nnz += (ku[p] + kl[p]);
  }


	/* Compute the number of rows optionally */
	if ( FirstBlockRow != NULL ) {
		*FirstBlockRow = 0;

		for(integer_t p = 0; p < CurrentPartition; p++)
			*FirstBlockRow += (ku[p] + kl[p]);
	}

	return (SPIKE_SUCCESS);
};


/*
  Adds the elemnts of a block to the reduced system according to the number of
  partition, and the block type.

  p = number of partition
  n = list of partition dimensions
  R = reduced system
  B = block (spike)
  type = block location flag
*/
Error_t matrix_FillReduced ( const integer_t TotalPartitions,
														 const integer_t CurrentPartition,
                             integer_t     *n,
                             integer_t     *ku,
                             integer_t     *kl,
                             matrix_t      *R,
                             block_t*       B )
{
	// local variables
	integer_t nnz, rows;
	integer_t p = CurrentPartition;

	// reduced system dimensions
	integer_t* nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);

	// initialize blocks
	GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

	/* ------------- top spike elements ---------------- */
#ifdef _DEBUG_MATRIX_
	fprintf(stderr, "\tTop part covers from %d to %d\n", nr[p], nr[p] + ku[p] );
#endif

	for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
		if ( p > 0 )// add Wi elements
			for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) {
				integer_t wi_row = row - nr[p];
				integer_t wi_col = col - (nr[p] - kl[p]);

#ifdef _DEBUG_MATRIX_
				fprintf(stderr, "\t\tAdding WI TOP element from row %d, col %d\n", wi_row, wi_col);
#endif

				if ( B->type == _W_BLOCK_ )
					R->aij[nnz++] = B->aij[wi_row*kl[p] + wi_col];
				else
					nnz++;
			}

		nnz++;// add diagonal element

		if ( p < (TotalPartitions - 1)) // add Vi elements
			for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) {
				integer_t vi_row = row - nr[p];
				integer_t vi_col = col - nr[p+1];
#ifdef _DEBUG_MATRIX_
				fprintf(stderr, "\t\tAdding VI TOP element from row %d, col %d\n", vi_row, vi_col);
#endif
				if ( B->type == _V_BLOCK_ )
					R->aij[nnz++] = B->aij[vi_row*kl[p] + vi_col];
				else
					nnz++;
			}
	}

	/* ------------- Bottom spike elements ---------------- */
#ifdef _DEBUG_MATRIX_
	fprintf(stderr, "\tBottom part covers from %d to %d\n", nr[p] +ku[p], nr[p+1] );
#endif

	for(integer_t row = (nr[p] + ku[p]); row < nr[p+1]; row++ ) {
		if ( p > 0 )// add Wi elements
			for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) {
				integer_t wi_row = (n[p+1] - n[p]) - kl[p] + (row - (nr[p] + ku[p]));
				integer_t wi_col = col - (nr[p] - kl[p]);

#ifdef _DEBUG_MATRIX_
				fprintf(stderr, "\t\tAdding WI BOTTOM element from row %d, col %d\n", wi_row, wi_col);
#endif
				if ( B->type == _W_BLOCK_ )
					R->aij[nnz++] = B->aij[wi_row*kl[p] + wi_col];
				else
					nnz++;
			}

		nnz++; // add diagonal element


		if ( p < (TotalPartitions - 1)) // add Vi elements
			for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) {
				integer_t vi_row = (n[p+1] -n[p]) - kl[p] + (row - (nr[p] + ku[p]));
				integer_t vi_col = col - nr[p+1];

#ifdef _DEBUG_MATRIX_
				fprintf(stderr, "\t\tAdding VI BOTTOM element from row %d, col %d\n", vi_row, vi_col);
#endif

				if (B->type == _V_BLOCK_)
					R->aij[nnz++] = B->aij[vi_row*kl[p] + vi_col];
				else
					nnz++;
			}
	}

	// clean up
	spike_nullify( nr );

  return (SPIKE_SUCCESS);
};

/*
  Inserts the elements of the block into the reduced linear system.

	TotalPartitions: total number of partitions in which the original system has been divided
	CurrentPartition: index of the current partition
	n*
	ku
	kl
	R: reduced system
	aij: pointer to the first element to insert.
	BlockType
	Location:
*/
/*
Error_t mpi_matrix_FillReduced (const integer_t TotalPartitions,
								const integer_t CurrentPartition,
								integer_t          *n,
								integer_t          *ku,
								integer_t          *kl,
								matrix_t           *R,
								complex_t          *aij,
								blocktype_t        BlockType,
								blocklocation_t    Location )
{
	// local variables
	integer_t nnz, rows;
	integer_t BlockAijCount = 0;
	integer_t p = CurrentPartition;

	// reduced system dimensions
	integer_t* nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);

	// initialize blocks
	GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

	// ------------- top spike elements ---------------- //
	if ( Location == _TOP_SECTION_ ) {
		for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
			if ( p > 0 && BlockType == _W_BLOCK_ )
				for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) 
					R->aij[nnz++] = aij[BlockAijCount++];
			else
				nnz += kl[p];
			
			nnz++;// add diagonal element

			if ( p < (TotalPartitions -1) &&  BlockType == _V_BLOCK_ )
				for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++)
					R->aij[nnz++] = aij[BlockAijCount++];
			else
				nnz += ku[p];
		}
	}

	// ------------- Bottom spike elements ---------------- //
	if ( Location == _BOTTOM_SECTION_ ) {
		for(integer_t row = (nr[p] + ku[p]); row < nr[p+1]; row++ ) {
			if ( p > 0  && BlockType == _W_BLOCK_ ) 
				for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++)
					R->aij[nnz++] = aij[BlockAijCount++];
			else
				nnz += kl[p];

			nnz++; // add diagonal element


			if ( p < (TotalPartitions -1) &&  BlockType == _V_BLOCK_ )
				for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++)
					R->aij[nnz++] = aij[BlockAijCount++];
			else
				nnz += ku[p];
		}
	}

	// clean up
	spike_nullify( nr );

  return (SPIKE_SUCCESS);
};
*/
Error_t mpi_matrix_FillReduced (const integer_t TotalPartitions,
								const integer_t CurrentPartition,
								integer_t          *n,
								integer_t          *ku,
								integer_t          *kl,
								matrix_t           *R,
								complex_t          *aij,
								blocktype_t        BlockType,
								blocklocation_t    Location )
{
	complex_t *restrict __attribute__ ((aligned (ALIGN_COMPLEX))) Raij = R->aij;
	complex_t *restrict __attribute__ ((aligned (ALIGN_COMPLEX))) Baij = aij;

	// local variables
	integer_t nnz, rows;
	integer_t BlockAijCount = 0;
	integer_t p = CurrentPartition;

	// reduced system dimensions
	integer_t* nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);

	// initialize blocks
	GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

	/* ------------- top spike elements ---------------- */
	for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
		if ( p > 0 )// add Wi elements
			if ( Location == _TOP_SECTION_ )
				if ( BlockType == _W_BLOCK_ )
					for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++)
						Raij[nnz++] = Baij[BlockAijCount++];
			else
				nnz += kl[p];
		else
			nnz += kl[p];
			

		nnz++;// add diagonal element

		if ( p < (TotalPartitions - 1)) // add Vi elements
			if ( Location == _TOP_SECTION_ )
				if ( BlockType == _V_BLOCK_ )
					for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++)
						Raij[nnz++] = Baij[BlockAijCount++];
				else
					nnz += ku[p];
			else
				nnz += ku[p];
	}

	/* ------------- Bottom spike elements ---------------- */
	for(integer_t row = (nr[p] + ku[p]); row < nr[p+1]; row++ ) {
		if ( p > 0 )// add Wi elements
			if ( Location == _BOTTOM_SECTION_ )
				if ( BlockType == _W_BLOCK_ )
					for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) 
						Raij[nnz++] = Baij[BlockAijCount++];
				else
					nnz += kl[p];
			else
				nnz += kl[p];
			

		nnz++; // add diagonal element


		if ( p < (TotalPartitions - 1)) // add Vi elements
			if ( Location == _BOTTOM_SECTION_ )
				if ( BlockType == _V_BLOCK_ )
					for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) 
						Raij[nnz++] = Baij[BlockAijCount++];
				else
					nnz += ku[p];
			else
				nnz += ku[p];
	}

	// clean up
	spike_nullify( nr );

  return (SPIKE_SUCCESS);
};