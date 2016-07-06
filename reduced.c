/* *
 * =====================================================================================
 *
 *       Filename:  reduced.c
 *
 *    Description:  This test checks the assembly of a reduced system from the
 *                  blocks coming from the solution of V and W blocks.
 *
 *        Version:  1.0
 *        Created:  05/07/16 09:59:35
 *       Revision:  None
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#include "spike_analysis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Returns a block initialized to a constant value.
 */
static block_t* block_Synthetic( const integer_t n, const integer_t m, const complex_t value )
{
  block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));

  B->n = n;
  B->m = m;
  B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, n * m, sizeof(complex_t));

  for (integer_t i = 0; i < n * m; i++)
    B->aij[i] = value;

  return (B);
}

static Error_t matrix_PrintAsDense( matrix_t* A, const char* msg)
{
  const integer_t nrows = A->n;
  const integer_t ncols = A->n;

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
      fprintf(stderr, "%.1f  ", D[row * ncols + col]);
    }
    fprintf(stderr, "\n");
  }

  spike_nullify(D);

  return (SPIKE_SUCCESS);
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
static Error_t ComputePrevNnzAndRows ( const integer_t p, integer_t* n, integer_t* ku, integer_t *kl, integer_t *nnz, integer_t *rows)
{
  *nnz = 0;
  *rows = 0;

  for(integer_t part = 0; part < p; part++){
    *nnz += n[part] * (ku[part] + kl[part] +1);
    *rows += n[part];
  }

  return (SPIKE_SUCCESS);
};

/*
  p = numero de particiones que lo originan
  k = (array) contiene el numero de columnas de cada bloque
  n = (array) contiene el numero de filas de cada bloque
 */
static matrix_t* matrix_Reduced( const integer_t p, integer_t *n, integer_t *ku, integer_t *kl )
{
  Error_t error;
  integer_t i,j,dim, nnz;

  ComputePrevNnzAndRows(p, n, ku, kl, &nnz, &dim);


  matrix_t* R = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));
  R->n        = dim;
  R->nnz      = nnz;
  R->colind   = (integer_t*) spike_malloc( ALIGN_INT    , R->nnz, sizeof(integer_t));
  R->rowptr   = (integer_t*) spike_malloc( ALIGN_INT    , R->n+1, sizeof(integer_t));
  R->aij      = (complex_t*) spike_malloc( ALIGN_COMPLEX, R->nnz, sizeof(complex_t));

  memset( (void*) R->colind, 0, (R->nnz) * sizeof(integer_t));
  memset( (void*) R->rowptr, 0, (R->n+1) * sizeof(integer_t));
  memset( (void*) R->aij   , 0, (R->nnz) * sizeof(complex_t));

  // initialize blocks
  for(integer_t part=0; part < p; part++){
    integer_t index;
    integer_t firstrow;
    integer_t firstcol;
    integer_t lastcol;

    // locate myself
    ComputePrevNnzAndRows(part, n, ku, kl, &index, &firstrow);

    // place elements
    for(integer_t row=firstrow; row < firstrow + n[part]; row++)
    {
      // add wi elements
      if ( kl[part]){
        firstcol = firstrow - kl[part];
        lastcol  = firstrow;

        for(integer_t col=firstcol; col < lastcol; col++) { R->colind[index++] = col; }
      }

      // add diagonal element
      R->colind[index] = row;
      R->aij[index]    = (complex_t) __unit;
      index++;

      // add vi elements
      if ( ku[part]){
        firstcol = firstrow + n[part];
        lastcol  = firstcol + ku[part];

        for(integer_t col=firstcol; col < lastcol; col++) { R->colind[index++] = col; }
      }

      // set rowptr properly
      R->rowptr[row+1] = R->rowptr[row] + (ku[part] + kl[part] +1);
    }
  }

  matrix_PrintAsDense(R, "Assembled reduced system");

  return (R);
}


int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nREDUCED SYSTEM ASSEMBLY TEST.\n\tINFO:"
                  "Creates a reduced system from the synthetic V and W blocks.");

	Error_t  res = 0;

  /* Create some synthetic spikes Vi, Wi */
  block_t *V0 = block_Synthetic( 4, 2, (complex_t) 1.0);
  block_t *W1 = block_Synthetic( 4, 1, (complex_t) 1.0);
  block_t *V1 = block_Synthetic( 4, 1, (complex_t) 2.0);
  block_t *W2 = block_Synthetic( 4, 1, (complex_t) 3.0);

  integer_t  p     = 3;
  integer_t  ku[3] = {2, 1, 0};
  integer_t  kl[3] = {0, 1, 1};
  integer_t  n [3] = {4, 4, 4};

  matrix_t* R = matrix_Reduced(p, n, ku, kl);


	block_Deallocate( V0 );
	block_Deallocate( W1 );
	block_Deallocate( V1 );
	block_Deallocate( W2 );
  matrix_Deallocate( R );

	fprintf(stderr, "\nTest result: PASSED.\n");

	fprintf(stderr, "\n");
	return 0;
}
