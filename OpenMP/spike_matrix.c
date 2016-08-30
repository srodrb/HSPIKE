#include "spike_matrix.h"
#include "spike_common.h"

/* -------------------------------------------------------------------- */
/* .. Functions affecting sparse CSR matrix structure                   */
/* -------------------------------------------------------------------- */

/*
	Loads a CSR matrix stored in our binary format specification.
*/
matrix_t* matrix_LoadCSR(const char* filename)
{
	/* local variables */
	integer_t dtype, nnz;

	/* allocate matrix structure */
	matrix_t* M = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));

	/* open file */
	FILE* f = spike_fopen( filename, "rb");

	/* load number of rows /columns */
	spike_fread( &M->n, sizeof(integer_t), 1, f );

	/* load number of nnz elements */
	spike_fread( &nnz, sizeof(integer_t), 1, f );

	M->nnz = (uLong_t) nnz;

	// TODO read data type, just to check everything is fine
	spike_fread( &dtype, sizeof(integer_t), 1, f );

	/* allocate space for matrix coefficients and load them */
	M->aij    = (complex_t*) spike_malloc( ALIGN_COMPLEX, M->nnz , sizeof(complex_t));
	spike_fread( M->aij, sizeof(complex_t), M->nnz, f );

	/* allocate space for matrix indices and load them */
	M->colind = (integer_t*) spike_malloc( ALIGN_INT, M->nnz , sizeof(integer_t));
	spike_fread( M->colind, sizeof(integer_t), M->nnz, f );

	/* allocate space for matrix row pointers and load them */
	M->rowptr = (integer_t*) spike_malloc( ALIGN_INT, M->n +1, sizeof(integer_t));
	spike_fread( M->rowptr, sizeof(integer_t), M->n + 1, f );

	/* clean up and resume */
	spike_fclose(f);

	fprintf(stderr, "Number of nnz elements %d, number of rows %d\n", M->nnz, M->n );

	return (M);
};

/*
	Save a CSR matrix stored in our binary format specification.
*/
void matrix_SaveCSR(const char* filename, matrix_t* M)
{
	/* local variables */
	integer_t dtype, nnz;

	/* open file */
	FILE* f = spike_fopen( filename, "wb");

	/* save number of rows /columns */
	spike_fwrite( &M->n, sizeof(integer_t), 1, f );

	/* save number of nnz elements */
	spike_fwrite( &M->nnz, sizeof(integer_t), 1, f );

	// TODO read data type, just to check everything is fine
	spike_fwrite( &M->type, sizeof(integer_t), 1, f );

	/* allocate space for matrix coefficients and save them */
	spike_fwrite( M->aij, sizeof(complex_t), M->nnz, f );

	/* allocate space for matrix indices and save them */
	spike_fwrite( M->colind, sizeof(integer_t), M->nnz, f );

	/* allocate space for matrix row pointers and save them */
	spike_fwrite( M->rowptr, sizeof(integer_t), M->n + 1, f );

	/* clean up and resume */
	spike_fclose(f);

};

/*
	Save a CSR matrix stored in our binary format specification.
*/
void block_SaveCSR(const char* filename, block_t* B)
{

	/* open file */
	FILE* f = spike_fopen( filename, "wb");

	/* save type of block */
	spike_fwrite( &B->type, sizeof(integer_t), 1, f );

	/* save section of block */
	spike_fwrite( &B->section, sizeof(integer_t), 1, f );

	/* save section of block */
	spike_fwrite( &B->n, sizeof(integer_t), 1, f );

	/* save section of block */
	spike_fwrite( &B->m, sizeof(integer_t), 1, f );

	/* save section of block */
	spike_fwrite( &B->ku, sizeof(integer_t), 1, f );

	/* save section of block */
	spike_fwrite( &B->kl, sizeof(integer_t), 1, f );

	/* allocate space for matrix coefficients and save them */
	spike_fwrite( B->aij, sizeof(complex_t), B->n*B->m, f );

	/* clean up and resume */
	spike_fclose(f);

};

/*
	Load a CSR matrix stored in our binary format specification.
*/
block_t* block_loadCSR(const char* filename)
{

	/* allocate matrix structure */
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));

	/* open file */
	FILE* f = spike_fopen( filename, "rb");

	/* Load block configuration */
	spike_fread( &B->type, 	  sizeof(integer_t), 1, f );
	spike_fread( &B->section, sizeof(integer_t), 1, f );
	spike_fread( &B->n, 	  sizeof(integer_t), 1, f );
	spike_fread( &B->m, 	  sizeof(integer_t), 1, f );
	spike_fread( &B->ku, 	  sizeof(integer_t), 1, f );
	spike_fread( &B->kl,	  sizeof(integer_t), 1, f );

	/* allocate space for block  data*/
	B->aij    = (complex_t*) spike_malloc( ALIGN_COMPLEX, B->n*B->m , sizeof(complex_t));

	/* load block data */
	spike_fread( B->aij, sizeof(complex_t), B->n*B->m, f );

	/* clean up and resume */
	spike_fclose(f);

	return B;

};

integer_t* vector_LoadPermutationArray(const integer_t n, const char* filename)
{
	FILE *f = spike_fopen( filename, "rb");

	integer_t *p = spike_malloc( ALIGN_INT, n, sizeof(integer_t));

	spike_fread( p, sizeof(integer_t), n, f);

	spike_fclose( f );

	return (p);
};

complex_t* vector_LoadRHS( const integer_t n, const char* filename )
{
	FILE *f = spike_fopen( filename, "rb");

	complex_t *p = spike_malloc( ALIGN_COMPLEX, n, sizeof(complex_t));

	spike_fread( p, sizeof(complex_t), n, f);

	spike_fclose( f );

	return (p);
};

/*
	Exports a matrix structure to a binary file.
 */
Error_t matrix_ExportBinary (matrix_t* M, const char* filename )
{
	/* local variables */
	integer_t dtype = 0, nnz;

	FILE* f = spike_fopen( filename, "wb");

	/* export number of rows/columns */
	spike_fwrite( &M->n, sizeof(integer_t), 1, f );

	/* export number of nnz elements */
	nnz = (int) M->nnz;
	spike_fwrite( &nnz, sizeof(integer_t), 1, f );

	/* export coefficient data type */
	spike_fwrite( &dtype, sizeof(integer_t), 1, f );

	/* export matrix coefficients */
	spike_fwrite( M->aij, sizeof(complex_t), M->nnz, f );

	/* export matrix indices */
	spike_fwrite( M->colind, sizeof(integer_t), M->nnz, f );

	/* export matrix row pointers */
	spike_fwrite( M->rowptr, sizeof(integer_t), M->n + 1, f );

	/* clean up and resume */
	spike_fclose(f);

	return (SPIKE_SUCCESS);
};


/*
	Creates an empty CSR sparse matriz of dimension n and nnz elements
*/
matrix_t* matrix_CreateEmptyMatrix( const integer_t n, const uLong_t nnz )
{
	matrix_t* R = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));
	R->n        = n;
	R->nnz      = nnz;
	R->ku       = 0;
	R->kl       = 0;
	R->colind   = (integer_t*) spike_malloc( ALIGN_INT    , R->nnz, sizeof(integer_t));
	R->rowptr   = (integer_t*) spike_malloc( ALIGN_INT    , R->n+1, sizeof(integer_t));
	R->aij      = (complex_t*) spike_malloc( ALIGN_COMPLEX, R->nnz, sizeof(complex_t));

	memset( (void*) R->colind, 0, (R->nnz) * sizeof(integer_t));
	memset( (void*) R->rowptr, 0, (R->n+1) * sizeof(integer_t));
	memset( (void*) R->aij   , 0, (R->nnz) * sizeof(complex_t));

	return (R);
};

/*
	This function receives the components of the matrix and creates a block structure.
	It is only intended to transform the input arguments of the solver interface to the
	structures that we use normally.
 */
matrix_t* matrix_CreateFromComponents(const integer_t n, 
	const uLong_t nnz, 
	integer_t *restrict colind, 
	integer_t *restrict rowptr, 
	complex_t *restrict aij)
{
	/* Allocate and fill matrix structure */
	matrix_t *A = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));
	A->n   = n;
	A->nnz = nnz;
	A->colind = colind;
	A->rowptr = rowptr;
	A->aij = aij;

	/* compute matrix bandwidth */
	matrix_ComputeBandwidth( A->n, A->colind, A->rowptr, A->aij, &A->ku, &A->kl );

	/* resume and return matrix */
	return (A);
};

Error_t matrix_Deallocate ( matrix_t* M )
{
	spike_nullify ( M->colind );
	spike_nullify ( M->rowptr );
	spike_nullify ( M->aij    );
	spike_nullify ( M );

	return (SPIKE_SUCCESS);
};

/*
 * returns 1 if both matrices are equal, 0 otherwise
 */
Bool_t matrix_AreEqual ( matrix_t* A, matrix_t* B )
{
	/* local variables */
	uLong_t i;

	/* function body */
	if( A->n != B->n )
	{
		fprintf(stderr, "\n%s: dimension mismatch", __FUNCTION__ );
		return (False);
	}

	if( A->nnz != B->nnz )
	{
		fprintf(stderr, "\n%s: number of nnz is not the same", __FUNCTION__ );
		return (False);
	}

	for (i = 0; i < A->nnz; i++)
	{
		if( number_IsEqual(A->aij[i],B->aij[i]) == False )
		{
			fprintf(stderr, "\n%s: "_I_"-th coefficents are not equal", __FUNCTION__, i);
			return (False);
		}
	}

	for (i = 0; i < A->nnz; i++)
	{
		if( A->colind[i] != B->colind[i]  )
		{
			fprintf(stderr, "\n%s: "_I_"-th indices are not equal", __FUNCTION__, i);
			return (False);
		}
	}

	for (i = 0; i < A->n+1; i++)
	{
		if( A->rowptr[i] != B->rowptr[i] )
		{
			fprintf(stderr, "\n%s: "_I_"-th row pointers are not equal", __FUNCTION__, i);
			return (False);
		}
	}

	return (True);
};

Error_t matrix_PrintAsSparse(matrix_t* M, const char* msg)
{
	/* local variables */
	uLong_t i = 0;

	/* function body */
	fprintf(stderr, "\n%s: %s", __FUNCTION__, msg);
	
	if ( M->n > _MAX_PRINT_DIMENSION_ ) {
		fprintf(stderr, "\n%s: Matrix is too large to print it!", __FUNCTION__ );
		return (SPIKE_SUCCESS);
	}


	fprintf(stderr, "\n\n\tMatrix dimension: "_I_", nnz: "_I_"\n", M->n, M->nnz);

	fprintf(stderr, "\n\n\tMatrix coefficients\n");
	for(i=0; i<M->nnz; i++){
		#ifndef _COMPLEX_ARITHMETIC_
			fprintf(stderr, "\t"_F_" ", M->aij[i] );
		#else
			fprintf(stderr, "\t"_F_","_F_"i ", M->aij[i].real, M->aij[i].imag );
		#endif
	}

	fprintf(stderr, "\n\n\tIndices\n");
	for(i=0; i<M->nnz; i++)
		fprintf(stderr, "\t"_I_" ", M->colind[i]);

	fprintf(stderr, "\n\n\tRow pointers\n");
	for(i=0; i<M->n +1; i++)
		fprintf(stderr, "\t"_I_" ", M->rowptr[i]);

	fprintf(stderr,"\n\n");

	return (SPIKE_SUCCESS);
};

Error_t matrix_PrintAsDense( matrix_t* A, const char* msg)
{
	const integer_t nrows = A->n;
	const integer_t ncols = A->n;

	if    (msg) { fprintf(stderr, "\n%s: %s\n\n", __FUNCTION__, msg);}
	else        { fprintf(stderr, "\n%s\n\n"    , __FUNCTION__ ); }

	if ( nrows > _MAX_PRINT_DIMENSION_ ) {
		fprintf(stderr, "\n%s: Matrix is too large to print it!", __FUNCTION__ );
		return (SPIKE_SUCCESS);
	}

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
			complex_t value = D[row * ncols + col];

			if ( number_IsLessThan ( value, __zero ) == True ){
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
				#endif
			}
			else{
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_" ", value.real, value.imag );
				#endif
			}

		}
		fprintf(stderr, "\n");
	}

	spike_nullify(D);

	return (SPIKE_SUCCESS);
};

/*
	Extracts a block from an sparse matrix.
	This function is intended to extract Wi and Vi blocks from the
	sparse given matrix M.
 */
block_t* matrix_ExtractBlock (  matrix_t* M,
								const integer_t r0,
								const integer_t rf,
								const integer_t c0,
								const integer_t cf,
								blocktype_t type )
{
	// TODO: extract the sparse sub-block, transpose it and insert it faster!
	integer_t row, col, idx;

	// allocates the -dense- block
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->type    = type;
	B->section = _WHOLE_SECTION_;
	B->n       = rf - r0;
	B->m       = cf - c0;

//	if ( M->ku <= 0 || M->kl <= 0 ){
//		fprintf(stderr, "\n%s: WARNING: Upper and Lower bandwidth are not determined for this matrix!", __FUNCTION__ );
//		
//		/* force the computetion of the matrix bandwidth */	
//		matrix_ComputeBandwidth( M->n, M->colind, M->rowptr, M->aij, &M->ku, &M->kl );		
//	}

	B->ku      = M->ku;
	B->kl      = M->kl;
	B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, ((size_t) B->n) * B->m, sizeof(complex_t));
	memset((void*) B->aij, 0, ((size_t) B->n) * B->m * sizeof(complex_t));

	// extract the elements, correct the indices and insert them into the dense block
	for(row=r0; row<rf; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];

			if ((col >= c0) && (col < cf)){
				B->aij[ (row -r0) + (col - c0)*B->n] = M->aij[idx]; // CSC
				//B->aij[ (row -r0) * B->m + (col - c0)] = M->aij[idx]; // CSR
			}
		}
	}

	return (B);
};

/*
	Impllementation of matrix_ExtractBlock function for the column
	blocking version.

	In addition to the reference function, it also receives a block to
	be used as buffer.

	Extracts a block from an sparse matrix.
	This function is intended to extract Wi and Vi blocks from the
	sparse given matrix M.
 */
Error_t matrix_ExtractBlock_blocking  (  matrix_t* M,
								block_t* B,
								const integer_t r0,
								const integer_t rf,
								const integer_t c0,
								const integer_t cf,
								blocktype_t type )
{
	// TODO: extract the sparse sub-block, transpose it and insert it faster!
	integer_t row, col, idx;

	// extract the elements, correct the indices and insert them into the dense block
	for(row=r0; row<rf; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];

			if ((col >= c0) && (col < cf)){
				B->aij[ (row -r0) + (col - c0)*B->n] = M->aij[idx]; // CSC
				//B->aij[ (row -r0) * B->m + (col - c0)] = M->aij[idx]; // CSR
			}
		}
	}

	return (SPIKE_SUCCESS);
};

/*
	It takes a small matrix (ej. Bi, Ci, etc) and extracts certain columns defined
	by c0,cf. The values on these columns are mapped over a dense block provided as
	input.


	Remark: in order to compute the starting row, correct values of
	B->ku and B->kl as well as B->type are needed.
 */
Error_t block_BuildBlockFromMatrix_blocking (  matrix_t* M,
								block_t* B,
								const integer_t r0,
								const integer_t rf,
								const integer_t c0,
								const integer_t cf,
								blocktype_t type )
{
	/* local variables */
	integer_t row, idx, col;


	/* check input dimensions */
	if ( (cf - c0) > B->m ) {
		fprintf(stderr, "\n%s: ERROR: dimension mismatch! B->m = %d, (cf-c0) = %d", __FUNCTION__, B->m, (cf - c0));
		abort();
	}

	// allocates the -dense- block

	/* compute starting row */
	integer_t firstRow = (B->type ==  _V_BLOCK_) ? B->n - B->ku : 0;
	integer_t lastRow  = (B->type ==  _V_BLOCK_) ? B->n         : B->kl;


	// extract the elements, correct the indices and insert them into the dense block
	for(row=0; row < M->n; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];
			
			/* place the element in the block, only if it is inside the [c0,cf] range */
			if ((col >= c0) && (col < cf)){
				printf("here with col:%d, c0:%d, cf:%d\n", col, c0, cf);	
				B->aij[ (col-c0) * B->n + (row + firstRow)] = M->aij[idx]; // CSC
				//B->aij[ (row -r0) * B->m + (col - c0)] = M->aij[idx]; // CSR
			}
		}
	}

	return (SPIKE_SUCCESS);
};


/*
	Extracts a sparse matrix sub-block from a given sparse matrix M.
 */
matrix_t* matrix_ExtractMatrix( matrix_t* M,
								const integer_t r0,
								const integer_t rf,
								const integer_t c0,
								const integer_t cf,
								matrixtype_t type)
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
	spike_timer_t tstart, tend;
	uLong_t nnz;
	integer_t rowind;
	integer_t nrows;
 	integer_t	idx;
 	integer_t	row;
 	integer_t	col;

 	/* initialize timer */
 	tstart = GetReferenceTime();

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
	matrix_t* B = matrix_CreateEmptyMatrix( rf - r0, nnz );
	B->type = type;

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
	tend = GetReferenceTime() - tstart;

//	fprintf(stderr, "\n%s: took %.6lf seconds.", __FUNCTION__, tend );

	/* compute the bandwidth of the sub-block */	
	matrix_ComputeBandwidth( B->n, B->colind, B->rowptr, B->aij, &B->ku, &B->kl );


	return (B);
};



/* -------------------------------------------------------------------- */
/* .. Functions affecting block structures.                             */
/* -------------------------------------------------------------------- */
/*
  p = numero de particiones que lo originan
  k = (array) contiene el numero de columnas de cada bloque
  n = (array) contiene el numero de filas de cada bloque
 */
matrix_t* matrix_CreateEmptyReducedSystem(const integer_t TotalPartitions, 
									integer_t *n, 
									integer_t *ku, 
									integer_t *kl )
{
	/* local variables */
	uLong_t nnz = 0;
	integer_t rows = 0;

	/* function body */

	// compute matrix dimensions and allocate the structure
	GetNnzAndRowsUpToPartition(TotalPartitions, TotalPartitions, ku, kl, &nnz, &rows );

	fprintf(stderr, "\n%s: statistics: nnz %zu, nrows %d \n", __FUNCTION__, nnz, rows );

	matrix_t* R = matrix_CreateEmptyMatrix( rows, nnz );

	// reduced system dimensions
	integer_t *nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);


  // initialize blocks
  for(integer_t p=0; p < TotalPartitions; p++)
	{
		GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

		/* ------------- top spike elements ---------------- */
		for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
			if ( p > 0 )// add Wi elements
			for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) {
				R->colind[nnz++] = col;
			}

			// add diagonal element
			R->colind[nnz  ] = row;
			R->aij   [nnz++] = (complex_t) __punit;

			// add Wi elements
			if ( p < (TotalPartitions - 1)) // add Vi elements
			for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) {
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
		for(integer_t row = (nr[p] + ku[p]); row < nr[p+1]; row++ ) {
			if ( p > 0 )// add Wi elements
			for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) {
				R->colind[nnz++] = col;
			}

			// add diagonal element
			R->colind[nnz  ] = row;
			R->aij   [nnz++] = (complex_t) __punit;

			// add Wi elements
			if ( p < (TotalPartitions - 1)) // add Vi elements
			for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) {
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

	/* clean up and resume */
	spike_nullify( nr );

 	return (R);
};

/*
	Creates an empty block of dimension n,m.
	It is intended to create buffers for system_solve call.
*/
block_t* block_CreateEmptyBlock (   const integer_t n, 
									const integer_t m, 
									const integer_t ku, 
									const integer_t kl, 
									blocktype_t type,
									blocksection_t section)
{
	/* local variables */
	size_t nmemb = ((size_t) n) * m;

	/* check for buffer overflow problems */
	if ( n < 0 || m < 0 || nmemb < 0 ) {
		fprintf(stderr, "\n%s: ERROR, n and m must be positive numbers (consider buffer overflow)\n", __FUNCTION__ );
		abort();
	}


	/* function body */
	block_t* B  = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->type     = type;
	B->section  = section;
	B->n        = n;
	B->m        = m;
	B->ku       = ku;
	B->kl       = kl;
	B->aij      = (complex_t*) spike_malloc( ALIGN_COMPLEX, nmemb, sizeof(complex_t));

	memset( (void*) B->aij, 0,  nmemb * sizeof(complex_t));

	return (B);
};

Error_t block_InitializeToValue( block_t* B, const complex_t value )
{
	complex_t *restrict aij = B->aij;

	for(integer_t i=0; i < (B->n * B->m); i++) aij[i] = value;

	return (SPIKE_SUCCESS);
};

/*
	This function receives the components of the vector and creates a block structure.
	It is only intended to transform the input arguments of the solver interface to the
	structures that we use normally.
 */
block_t* block_CreateFromComponents(const integer_t n, const integer_t m, complex_t *restrict Bij)
{
	/* Allocate and fill block structure */
	block_t *B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->n       = n;
	B->m       = m;
	B->aij     = Bij;
	B->ku      = 0;
	B->kl      = 0;
	B->type    = _RHS_BLOCK_;
	B->section = _WHOLE_SECTION_;

	/* resume and return block */
	return (B);
};

/*
	This functions receives a sparse matrix and build inserts it into a dense block.
	The block is returned.

	It is intended to build the Bi / Ci dense blocks, needed to solve the linear system
	Ai * Vi = Bi from the Bi/Ci sparse sub-matrices.  
 */
block_t* block_BuildBlockFromMatrix( matrix_t *M, blocktype_t type,
	const integer_t n, const integer_t m, const integer_t ku, const integer_t kl )
{
	/* local variables */
	integer_t row, idx, col;

	// allocates the -dense- block
	block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));
	B->type    = type;
	B->section = _WHOLE_SECTION_;
	B->n       = n;
	B->m       = (B->type ==  _V_BLOCK_) ? ku : kl;
	B->ku      = ku;
	B->kl      = kl;

	if ( B->ku <= 0 || B->kl <= 0 ){
		fprintf(stderr, "\n%s: ERROR: Upper and Lower bandwidth are not determined for this matrix!", __FUNCTION__ );		
		abort();
	}


	B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, B->n * B->m, sizeof(complex_t));
	memset((void*) B->aij, 0, B->n * B->m * sizeof(complex_t));


	/* locate myself */
	integer_t firstRow = (B->type ==  _V_BLOCK_) ? B->n - B->ku : 0;
	integer_t lastRow  = (B->type ==  _V_BLOCK_) ? B->n         : B->kl;


	// extract the elements, correct the indices and insert them into the dense block
	for(row=0; row < M->n; row++)
	{
		for(idx=M->rowptr[row]; idx<M->rowptr[row+1]; idx++)
		{
			col = M->colind[idx];
			
			/* place the element in the block */	
			B->aij[ col * B->n + (row + firstRow)] = M->aij[idx]; // CSC
			//B->aij[ (row -r0) * B->m + (col - c0)] = M->aij[idx]; // CSR
		}
	}


	matrix_Deallocate(M);

	return (B);
};

Error_t block_Print( block_t* B, const char* msg )
{

	const integer_t nrows = B->n;
	const integer_t ncols = B->m;

	if ( msg ) fprintf(stderr, "\n%s:%s\n\t", __FUNCTION__, msg);
	else       fprintf(stderr, "\n%s\n\t"   , __FUNCTION__     );

	if ( nrows > _MAX_PRINT_DIMENSION_ || ncols > _MAX_PRINT_DIMENSION_ ) {
		fprintf(stderr, "\n%s: Block is too large to print it!", __FUNCTION__ );
		return (SPIKE_SUCCESS);
	}

	fprintf(stderr, "\nBlock info: dimension ("_I_","_I_"), bandwidth ("_I_","_I_")\n\n\t", 
					B->n, B->m, B->ku, B->kl);

	for(integer_t row=0; row < nrows; row++){
		for(integer_t col=0; col < ncols; col++) {
			complex_t value = B->aij[row + nrows*col];
			
			if ( number_IsLessThan ( value, __zero ) == True ){
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
				#endif
			}
			else{
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_" ", value.real, value.imag );
				#endif
			}
		}
		fprintf(stderr, "\n\t");
	}

	fprintf(stderr, "\n");

	return (SPIKE_SUCCESS);
};


/* deallocates a block structure */
Error_t block_Deallocate(block_t* B)
{
	spike_nullify( B->aij );
	spike_nullify( B );

	return (SPIKE_SUCCESS);
};

/*
 * returns True if both block are equal, False otherwise
 */
Bool_t block_AreEqual( block_t* A, block_t* B )
{
	if( A->n != B->n || A->m != B->m)
	{
		fprintf(stderr, "\n%s: dimension mismatch", __FUNCTION__ );
		return (False);
	}

	if( A->ku != B->ku || A->kl != B->kl)
	{
		fprintf(stderr, "\n%s: dandwidth mismatch", __FUNCTION__ );
		return (False);
	}

	for (integer_t i = 0; i < A->n * A->m; i++)
	{
		if( number_IsEqual( A->aij[i], B->aij[i]) == False )
		{
			fprintf(stderr, "\n"_I_"-th coefficents are not equal", i);
			return (False);
		}
	}

	return (True);
};



/*
	Transposes the block elements, it used to switch from column-major ordering
	to row-major ordering.

	In particular, this function is used after the extraction of the Vi / Wi
	tips of the solution vector of the reduced systems. These blocks need to be
	converted to row-major ordering in order to add them to the reduced system
	efficiently.

	When the Intel MKL backend is used, it calls mkl_?imatcopy() function.
	https://software.intel.com/es-es/node/468654#60DD96D3-068C-4AC9-8CE2-D84302F1DCED

	Be aware that it is only needed to change the memory layout, not finding the
	transpose of the matrix in the math sense. (i.e., conjugate transpose == transpose)
*/
Error_t block_Transpose( block_t* B )
{
	// TODO omatcopy2 and omatcopy are the out-of-place version of the transpose
	// matrix operation and seems to achieve much higher throughput. Would be a good
	// idea testing them.
	// See "In-place transposition of rectangular matrices on accelerators (H.Whu)"

	/* transpose the block using mkl_?imatcopy */
	#ifdef __INTEL_MKL__
		CALL_LA_KERNEL(mkl_,_PPREF_,imatcopy) ('R', 'T', B->n, B->n, __punit, B->aij, B->n, B->n );
	#else
		fprintf(stderr, "%s: NOT IMPLEMENTED\n", __FUNCTION__);
		abort();
	#endif


	return (SPIKE_SUCCESS);
};

Error_t block_Transpose_blocking( complex_t* aij, const integer_t n, const integer_t m )
{
	// TODO omatcopy2 and omatcopy are the out-of-place version of the transpose
	// matrix operation and seems to achieve much higher throughput. Would be a good
	// idea testing them.
	// See "In-place transposition of rectangular matrices on accelerators (H.Whu)"

	/* transpose the block using mkl_?imatcopy */
	#ifdef __INTEL_MKL__
		CALL_LA_KERNEL(mkl_,_PPREF_,imatcopy) ('C', 'T', n, m, __punit, aij, n, m );
//		CALL_LA_KERNEL(mkl_,_PPREF_,imatcopy) ('C', 'T', 3, 2, __punit, aij, 3, 2 );
	#else
		fprintf(stderr, "%s: NOT IMPLEMENTED\n", __FUNCTION__);
		abort();
	#endif


	return (SPIKE_SUCCESS);
};

/*
	Extracts a section of the block and returns it as a block. 

	The section to extract is specified by the "section" argument.
	The memory layout of the output block is specified by the "layout" argument.

	The function assumes that the input block is always stored in column-major ordering,
	so there is no need to transpose the output block if the output block is requiered
	to be in column-major ordering 
 */
block_t* block_ExtractTip ( block_t* B, blocksection_t section, memlayout_t layout )
{
  block_t*  SubBlock;
  integer_t RowOffset;
  integer_t ChunkSize;

  switch ( section ){
  	case _TOP_SECTION_ : {
	    RowOffset = 0;
	    ChunkSize = B->ku;
  		break;
  	}
  	case _BOTTOM_SECTION_ : {
	    RowOffset = B->n - B->kl;
	    ChunkSize = B->kl;
  		break;
  	}
  	case _CENTRAL_SECTION_ : {
	    RowOffset = B->ku;
	    ChunkSize = B->n - (B->ku + B->kl); 
  		break;
  	}
  	default: {
  		fprintf(stderr, "\n%s: ERROR: unrecognized block section value", __FUNCTION__ );
  		abort();
  	}
  } /* end of switch statement */
  
  SubBlock = block_CreateEmptyBlock( ChunkSize, B->m, B->ku, B->kl, B->type, section );
  
  for(integer_t col=0; col < SubBlock->m; col++)
    memcpy((void*) &SubBlock->aij[col * ChunkSize], (const void*) &B->aij[col*B->n + RowOffset], ChunkSize * sizeof(complex_t));


  /* 
  	central section of the block is never transposed, since it is only
	used as RHS of a linear system and sparse direct solvers require
	column-major ordering.
	It does not make sense to transpose a column vector neither.
  */
  if ( layout == _ROWMAJOR_ && B->m > 1 ) 
  	block_Transpose( SubBlock );
  
  return (SubBlock);    
};

/*
	This is an extension of the block_ExtractTip function. Now, it receives a buffer
	and copies to it a section of the input block (src) determined by the section
	flag and the columns interval.

	The values of src are copied to dst starting at dst->aij[0].
 */
Error_t block_ExtractTip_blocking ( block_t *dst,
									block_t *src, 
									const integer_t c0,
									const integer_t cf,
									blocksection_t section,
									memlayout_t layout )

{
  	integer_t RowOffset; /* row offset on the src block */
  	integer_t ChunkSize; /* chunksize to copy on the dst block */
	integer_t col; 

	/* add the section flag to the block. In this way, the routine */
	/* that incorporates it to the reduced RHS is able to read the */
	/* updated flag 											   */
	dst->section = section;

  	switch ( section ){
	  	case _TOP_SECTION_ : {
		    RowOffset = 0;
		    ChunkSize = src->ku;
	  		break;
	  	}
	  	case _BOTTOM_SECTION_ : {
		    RowOffset = src->n - src->kl;
		    ChunkSize = src->kl;
	  		break;
	  	}
	  	case _CENTRAL_SECTION_ : {
		    RowOffset = src->ku;
		    ChunkSize = src->n - (src->ku + src->kl); 
	  		break;
	  	}
	  	default: {
	  		fprintf(stderr, "\n%s: ERROR: unrecognized block section value", __FUNCTION__ );
	  		abort();
  	}
  } /* end of switch statement */
  
  for(col=c0; col < cf; col++)
  	memcpy((void*) &dst->aij[col * ChunkSize], (const void*) &src->aij[(col)*src->n + RowOffset], ChunkSize * sizeof(complex_t));

    // memcpy((void*) &dst->aij[col * ChunkSize], (const void*) &src->aij[col*src->n], ChunkSize * sizeof(complex_t));

  /* 
  	central section of the block is never transposed, since it is only
	used as RHS of a linear system and sparse direct solvers require
	column-major ordering.
	It does not make sense to transpose a column vector neither.
  */
  if ( layout == _ROWMAJOR_ && src->m > 1 ) 
  	block_Transpose_blocking( dst->aij, ChunkSize, cf - c0 );
  
  return (SPIKE_SUCCESS);    
};

/*
	This is an extension of the block_ExtractTip function. Now, it receives a buffer
	and copies to it a section of the input block (src) determined by the section
	flag and the columns interval.

	The values of src are copied to dst starting at dst->aij[colOffset].
 */
Error_t block_ExtractTip_blocking_mpi ( block_t *dst,
									block_t *src, 
									const integer_t c0,
									const integer_t cf,
									const integer_t colOffset,
									blocksection_t section,
									memlayout_t layout )

{
  	integer_t RowOffset; /* row offset on the src block */
  	integer_t ChunkSize; /* chunksize to copy on the dst block */
	integer_t col; 

	/* add the section flag to the block. In this way, the routine */
	/* that incorporates it to the reduced RHS is able to read the */
	/* updated flag 											   */
	dst->section = section;

  	switch ( section ){
	  	case _TOP_SECTION_ : {
		    RowOffset = 0;
		    ChunkSize = src->ku;
	  		break;
	  	}
	  	case _BOTTOM_SECTION_ : {
		    RowOffset = src->n - src->kl;
		    ChunkSize = src->kl;
	  		break;
	  	}
	  	case _CENTRAL_SECTION_ : {
		    RowOffset = src->ku;
		    ChunkSize = src->n - (src->ku + src->kl); 
	  		break;
	  	}
	  	default: {
	  		fprintf(stderr, "\n%s: ERROR: unrecognized block section value", __FUNCTION__ );
	  		abort();
  	}
  } /* end of switch statement */
  
  for(col=c0; col < cf; col++)
  	memcpy((void*) &dst->aij[(col+colOffset) * ChunkSize], (const void*) &src->aij[(col)*src->n + RowOffset], ChunkSize * sizeof(complex_t));

    // memcpy((void*) &dst->aij[col * ChunkSize], (const void*) &src->aij[col*src->n], ChunkSize * sizeof(complex_t));

  /* 
  	central section of the block is never transposed, since it is only
	used as RHS of a linear system and sparse direct solvers require
	column-major ordering.
	It does not make sense to transpose a column vector neither.
  */
  if ( layout == _ROWMAJOR_ && src->m > 1 ) 
  	block_Transpose_blocking( dst->aij, ChunkSize, cf - c0 );
  
  return (SPIKE_SUCCESS);    
};


/*
	Extracts a part of a given block.
	The extracted block starts at the n0-th row and ends at the nf-th row.
	Since we are not interested in extracting certain columns of a block, it goes
	through all the width of the block.
 */
block_t* block_ExtractBlock (block_t* B, const integer_t n0, const integer_t nf )
{
	block_t* SubBlock = block_CreateEmptyBlock( nf - n0, B->m, 0, 0, B->type, _WHOLE_SECTION_ );

	/* copy the elements from the reference block copying them to the subblock */
	for(integer_t col=0; col < SubBlock->m; col++)
    	memcpy((void*) &SubBlock->aij[col * SubBlock->n], (const void*) &B->aij[col*B->n + n0], SubBlock->n * sizeof(complex_t));


	return (SubBlock);
};

/*
	Low-level implementation of the block_ExtractBlock routine.

	It uses a buffer for storing the coefficients and limits the block
	also on the column direction.

	Extracts a part of a given block.
	The extracted block starts at the n0-th row and ends at the nf-th row.
	Since we are not interested in extracting certain columns of a block, it goes
	through all the width of the block.
 */
Error_t block_ExtractBlock_blocking   ( block_t *dst, 
										block_t *src, 
										const integer_t n0, 
										const integer_t nf,
										const integer_t c0,
										const integer_t cf )
{
	/* local variables */
	integer_t col = 0;
	integer_t nrows = nf - n0;

	/* copy the elements from the reference block copying them to the subblock */
	for(col=c0; col < cf; col++)
    	memcpy((void*) &dst->aij[ (col-c0) * nrows], (const void*) &src->aij[col*src->n + n0], nrows * sizeof(complex_t));


	return (SPIKE_SUCCESS);
};

/*
	Sets the values of upper (kl) and lower (kl) bandwidth to a block.

	Some times it is not possible to specify these values at the moment of the block
	creation, for example when partitioning f into fi blocks, because the bandwidth
	of the matrix is not known until the matrix goes through the analysis process.
	Hence, this function it used to set these values.
 */
Error_t block_SetBandwidthValues  (block_t* B, const integer_t ku, const integer_t kl)
{
	B->ku = ku;
	B->kl = kl;

	return (SPIKE_SUCCESS);
};

/*
	Creates a reduced RHS for the reduced sytem.
 */
block_t* block_CreateReducedRHS (const integer_t TotalPartitions,
								integer_t *ku,
								integer_t *kl,
								const integer_t nrhs)
{
	integer_t nrows = 0;

	for(integer_t i=0; i < TotalPartitions; i++) nrows += (ku[i] + kl[i]);

	block_t* B = block_CreateEmptyBlock( nrows, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_);

	return (B);
};

/*
	Adds a block to the reduced system RHS. It is intended to
	incorporate the tips of the RHS to the reduced RHS.
 */
Error_t block_AddTipTOReducedRHS   (const integer_t CurrentPartition,
									integer_t          *ku,
									integer_t          *kl,
									block_t            *RHS,
									block_t            *B)
{
	/* check dimensions */
	if ( RHS->m != B->m ) {
		fprintf(stderr, "Dimension mismatch!\n");
		abort();
	}

	/* compute the row at which the previous parition ends */
	integer_t row = 0;

	for(integer_t i=0; i < CurrentPartition; i++ ) row += (ku[i] + kl[i]);

	/* for bottom tips, we have to increase further the counter */
	if ( B->section == _BOTTOM_SECTION_ ) row += ku[CurrentPartition];

	/* copy the elements from the reference block copying them to the subblock */
	for(integer_t col=0; col < B->m; col++)
     	memcpy((void*) &RHS->aij[col * RHS->n + row], (const void*) &B->aij[col*B->n], B->n * sizeof(complex_t));

	return (SPIKE_SUCCESS);
};

/*
	Blocking version of the block_AddTipTOReducedRHS function.

	It incorporates the range of columns to the original function.

	Adds a block to the reduced system RHS. It is intended to
	incorporate the tips of the RHS to the reduced RHS.
 */
Error_t block_AddTipTOReducedRHS_blocking   (const integer_t CurrentPartition,
									const integer_t     c0,
									const integer_t     cf,
									integer_t          *ku,
									integer_t          *kl,
									block_t            *RHS,
									block_t            *B)
{
	/* local variables */
	integer_t col = 0, row = 0, i=0, chunksize;

	/* how many elements should I copy */
	chunksize = ( B->section == _TOP_SECTION_ ) ? B->ku : B->kl;

	/* compute the row at which the previous parition ends */
	for( i=0; i < CurrentPartition; i++ ) row += (ku[i] + kl[i]);

	/* for bottom tips, we have to increase further the counter */
	if ( B->section == _BOTTOM_SECTION_ ) row += ku[CurrentPartition];

	/* copy the elements from the reference block copying them to the subblock */
	for(col=c0; col < cf; col++)
     	memcpy((void*) &RHS->aij[col * RHS->n + row], (const void*) &B->aij[ (col-c0) * chunksize], chunksize * sizeof(complex_t));

	return (SPIKE_SUCCESS);
};

/*
	This function inserts a block into another block, both in column-major layout.

	It is intended to insert the solution xi into the RHS x at the end of the algorithm.
 */
Error_t block_AddBlockToRHS (block_t *x, block_t* xi, const integer_t n0, const integer_t nf)
{
	for(integer_t col=0; col < x->m; col++)
     	memcpy((void*) &x->aij[col * x->n + n0], (const void*) xi->aij, (nf - n0) * sizeof(complex_t));

	return (SPIKE_SUCCESS);
};

/* -------------------------------------------------------------------- */
/* .. Functions for reduced sytem assembly.                             */
/* -------------------------------------------------------------------- */


/*
	Computes an array, similar to R->n that gives the number of rows per block.
 */
static integer_t* ComputeReducedSytemDimensions(integer_t partitions, 
												integer_t *ku, 
												integer_t *kl)
{
	integer_t* nr = (integer_t*) spike_malloc( ALIGN_INT, partitions + 1, sizeof(integer_t));
	memset( (void*) nr, 0, (partitions +1) * sizeof(integer_t));

	for(integer_t i=0; i < partitions; i++) {
		nr[i+1] = nr[i] + (ku[i] + kl[i]);
	}

	return (nr);
};

/*
  Computes the number of nnz in the reduced system.

  p = total number of partitions
  n = number of rows per block (array)
  ku = upper bandwidth for Vi blocks
  kl = lower bandwidth for Wi blocks
  nnz = number of nnz elements
  dim = total number of rows in the reduced system
*/
static Error_t    GetNnzAndRowsUpToPartition   (const integer_t TotalPartitions, 
												const integer_t CurrentPartition, 
												integer_t *ku,
												integer_t *kl, 
												uLong_t *nnz, 
												integer_t *FirstBlockRow )
{
	/* Compute the number of nnz elements up to the actual partition */
  *nnz      = 0;
  for(integer_t p = 0; p < CurrentPartition; p++)
  {
		if ( p == 0 ){
			*nnz = ku[p] * (ku[p] + kl[p]);
		}
		else if ( p == (TotalPartitions -1)) {
			*nnz += kl[p] * (ku[p] + kl[p]);
		}
		else{
			*nnz += kl[p] * (ku[p] + kl[p]) + ku[p] * (ku[p] + kl[p]);
		}

		/* add diagonal elements */
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

Error_t matrix_AddTipToReducedMatrix (const integer_t TotalPartitions,
										const integer_t CurrentPartition,
										integer_t          *n,
										integer_t          *ku,
										integer_t          *kl,
										matrix_t           *R,
										block_t            *B)
{
	complex_t *restrict __attribute__ ((aligned (ALIGN_COMPLEX))) Raij = R->aij;
	complex_t *restrict __attribute__ ((aligned (ALIGN_COMPLEX))) Baij = B->aij;


	// local variables
	uLong_t   nnz;
	integer_t rows;
	integer_t BlockAijCount = 0;
	integer_t p = CurrentPartition;

	// reduced system dimensions
	integer_t* nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);

	// initialize blocks
	GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

	/* ------------- top spike elements ---------------- */
	for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
		if ( p > 0 )// add Wi elements
			if ( B->section == _TOP_SECTION_ )
				if ( B->type == _W_BLOCK_ )
					for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++)
						Raij[nnz++] = Baij[BlockAijCount++];
			else
				nnz += kl[p];
		else
			nnz += kl[p];
			

		nnz++;// add diagonal element

		if ( p < (TotalPartitions - 1)) // add Vi elements
			if ( B->section == _TOP_SECTION_ )
				if ( B->type == _V_BLOCK_ )
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
			if ( B->section == _BOTTOM_SECTION_ )
				if ( B->type == _W_BLOCK_ )
					for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) 
						Raij[nnz++] = Baij[BlockAijCount++];
				else
					nnz += kl[p];
			else
				nnz += kl[p];
			

		nnz++; // add diagonal element


		if ( p < (TotalPartitions - 1)) // add Vi elements
			if ( B->section == _BOTTOM_SECTION_ )
				if ( B->type == _V_BLOCK_ )
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

/*
	This is the low-lelvel implementation of the matrix_AddTipToReducedMatrix
	function.

	This function implements a blocking strategy not only on the row direction
	but on the column direction also. Hence, it needs to know the column interval
	of the blocking.
 */
Error_t matrix_AddTipToReducedMatrix_blocking (const integer_t TotalPartitions,
										const integer_t CurrentPartition,
										const integer_t    c0,
										const integer_t    cf,
										integer_t          *n,
										integer_t          *ku,
										integer_t          *kl,
										matrix_t           *R,
										block_t            *B)
{
	complex_t *restrict __attribute__ ((aligned (ALIGN_COMPLEX))) Raij = R->aij;
	complex_t *restrict __attribute__ ((aligned (ALIGN_COMPLEX))) Baij = B->aij;


	// local variables
	uLong_t   nnz;
	integer_t rows;
	integer_t BlockAijCount = 0;
	integer_t p = CurrentPartition;

	// reduced system dimensions
	integer_t* nr = ComputeReducedSytemDimensions( TotalPartitions, ku, kl);

	// initialize blocks
	GetNnzAndRowsUpToPartition(TotalPartitions, p, ku, kl, &nnz, NULL );

	/* compute the number of columns of the blocking buffer */
	integer_t colblock = cf - c0;

	/* ------------- top spike elements ---------------- */
	for(integer_t row = nr[p]; row < (nr[p] + ku[p]); row++ ) {
		if ( p > 0 )// add Wi elements
			if ( B->section == _TOP_SECTION_ )
				if ( B->type == _W_BLOCK_ )
					for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++)
						if ( col >= (nr[p] - kl[p]) + c0 && col < (nr[p] - kl[p]) + c0 + colblock ){
							Raij[nnz++] = Baij[BlockAijCount++];
						}
						else{
							nnz           += 1;
						}
			else
				nnz += kl[p];
		else
			nnz += kl[p];
			

		nnz++;// add diagonal element

		if ( p < (TotalPartitions - 1)) // add Vi elements
			if ( B->section == _TOP_SECTION_ )
				if ( B->type == _V_BLOCK_ )
					for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++)
						if ( col >= (nr[p+1] + c0) && col < (nr[p+1] + c0 + colblock) ){
							Raij[nnz++] = Baij[BlockAijCount++];
						}
						else{
							nnz           += 1;
						}
				else
					nnz += ku[p];
			else
				nnz += ku[p];
	}

	/* ------------- Bottom spike elements ---------------- */
	for(integer_t row = (nr[p] + ku[p]); row < nr[p+1]; row++ ) {
		if ( p > 0 )// add Wi elements
			if ( B->section == _BOTTOM_SECTION_ )
				if ( B->type == _W_BLOCK_ )
					for(integer_t col= (nr[p] - kl[p]); col < nr[p]; col++) 
						if ( col >= (nr[p] - kl[p]) + c0 && col < (nr[p] - kl[p]) + c0 + colblock ){
							Raij[nnz++] = Baij[BlockAijCount++];
						}
						else{
							nnz           += 1;
						}
				else
					nnz += kl[p];
			else
				nnz += kl[p];
			

		nnz++; // add diagonal element


		if ( p < (TotalPartitions - 1)) // add Vi elements
			if ( B->section == _BOTTOM_SECTION_ )
				if ( B->type == _V_BLOCK_ )
					for(integer_t col= nr[p+1]; col < (nr[p+1] + ku[p]); col++) 
						if ( col >= (nr[p+1] + c0) && col < (nr[p+1] + c0 + colblock) ){
							Raij[nnz++] = Baij[BlockAijCount++];
						}
						else{
							nnz           += 1;
						}
				else
					nnz += ku[p];
			else
				nnz += ku[p];
	}

	// clean up
	spike_nullify( nr );

  return (SPIKE_SUCCESS);
};

Error_t reduced_PrintAsDense (matrix_t *R, block_t *X, block_t *Y, const char* msg)
{
	/* local variables */
	Bool_t displayDots = False; /* only if there are too many rhs to print */
	integer_t nrhs     = 0;

	/* check correctness of dimensions */
	if ( R->n != Y->n || R->n != Y->m ){
		if ( X != NULL && (Y->n != X->n || Y->m != X->m ))
		{
		fprintf(stderr, "Dimension mismatch\n");
		abort();
		}
	}

	/* control the max. number of rhs for print */
	if( Y->m > _MAX_PRINT_RHS_ ){
		nrhs = _MAX_PRINT_RHS_;
		displayDots = True;
	}
	else{
		nrhs = Y->m;
	}

	if    (msg) { fprintf(stderr, "\n%s: %s\n\n", __FUNCTION__, msg);}
	else        { fprintf(stderr, "\n%s\n\n"    , __FUNCTION__ ); }

	if ( R->n > _MAX_PRINT_DIMENSION_ ) {
		fprintf(stderr, "\n%s: Matrix is too large to print it!", __FUNCTION__ );
		return (SPIKE_SUCCESS);
	}

	fprintf(stderr, "\nMatrix dimension %d, nrhs %d\n", R->n, Y->m);

	/* first, we need to convert the sparse matrix R to a dense representation */
	complex_t *D = (complex_t*) spike_malloc( ALIGN_COMPLEX, R->n * R->n, sizeof(complex_t));
	memset( (void*) D, 0, R->n * R->n * sizeof(complex_t));

	for(integer_t row = 0; row < R->n; row++){
		for(integer_t idx = R->rowptr[row]; idx < R->rowptr[row+1]; idx++ ){
			integer_t col = R->colind[idx];
			D[ row * R->n + col] = R->aij[idx];
		}
	}

	/* display the system nicely */
	for(integer_t row = 0; row < R->n; row++){
		/* print coefficient matrix */ 
		for(integer_t col = 0; col < R->n; col++){
			complex_t value = D[row * R->n + col];

			if ( number_IsLessThan ( value, __zero ) == True ){
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
				#endif
			}
			else{
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
				#endif
			}
		}

		fprintf(stderr, "\t");

	
		/* print x block, if needed */
		if ( X != NULL ){
			for(integer_t col=0; col < nrhs; col++) {
				complex_t value = X->aij[row + X->n*col];
	
				if ( number_IsLessThan( value, __zero )){
					#ifndef _COMPLEX_ARITHMETIC_
						fprintf(stderr, "\t"_F_" ", value );
					#else
						fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
					#endif
				}
				else{
					#ifndef _COMPLEX_ARITHMETIC_
						fprintf(stderr, "\t"_F_" ", value );
					#else
						fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
					#endif
				}
			}
			
			if ( displayDots ) {fprintf(stderr, " ..."); }

			fprintf(stderr, "\t");
		}

		/* print y block */
		for(integer_t col=0; col < nrhs; col++) {
			complex_t value = Y->aij[row + Y->n*col];

			if ( number_IsLessThan( value, __zero )){
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
				#endif
			}
			else{
				#ifndef _COMPLEX_ARITHMETIC_
					fprintf(stderr, "\t"_F_" ", value );
				#else
					fprintf(stderr, "\t"_F_","_F_"i ", value.real, value.imag );
				#endif
			}
		}

		if ( displayDots ) {fprintf(stderr, " ..."); }
		
		fprintf(stderr, "\n");
	}	



	/* clean up and resume */
	spike_nullify( D );

	return (SPIKE_SUCCESS);
};
