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

/*
  p = numero de particiones que lo originan
  k = (array) contiene el numero de columnas de cada bloque
  n = (array) contiene el numero de filas de cada bloque
 */
static matrix_t* matrix_Reduced( const integer_t p, integer_t *n, integer_t *k )
{
  integer_t i,j,partition,row,col;

  integer_t dim = 0; // reduced system dimension
  integer_t nnz = 0; // number of nnz in the reduced system

  for(i=0; i<p; i++) dim += n[i];
  for(i=0; i<p; i++) nnz += (n[i] * k[i]) + dim;

  fprintf(stderr, "Reduced system dimension is %d, nnz %d\n", dim, nnz);

  matrix_t* R = (matrix_t*) spike_malloc( ALIGN_INT, 1, sizeof(matrix_t));
  R->n = dim;
  R->nnz = nnz;

  R->colind = (integer_t*) spike_malloc( ALIGN_INT    , R->nnz, sizeof(integer_t));
  R->rowptr = (integer_t*) spike_malloc( ALIGN_INT    , R->n+1, sizeof(integer_t));
  R->aij    = (complex_t*) spike_malloc( ALIGN_COMPLEX, R->nnz, sizeof(complex_t));

  R->rowptr[0] = 0;

  // initialize first partition
  for(row=0; row < n[0]; row++)
  {
    integer_t diagidx = k[0] * row + row;

    R->aij   [diagidx] = (complex_t) 1.0;
    R->colind[diagidx] = row;

    for(col=1; col < (k[0]+1); col++)
      R->colind[diagidx + col] = row + col;

    R->rowptr[row+1] = R->rowptr[row] + k[0] + 1;
  }

  // initialize middle blocks
  for(partition=1; partition < (p-1); partition++){
    integer_t startrow = 0;

    for(i=0; i < partition; i++) startrow += n[i];

    fprintf(stderr, "%d-th partition starts at %d-th row\n", partition, startrow);

    for(row=startrow; row < (startrow + n[partition]); row++){

    }
  }



  matrix_Print( R, "Reduced system");

  return (R);
}


int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nREDUCED SYSTEM ASSEMBLY TEST.\n\tINFO: \
                  Creates a reduced system from the synthetic V and W blocks.");

	Error_t  res = 0;

  /* Create some synthetic spikes Vi, Wi */
  block_t *V0 = block_Synthetic( 4, 1, (complex_t) 1.0);
  block_t *W1 = block_Synthetic( 4, 1, (complex_t) 1.0);
  block_t *V1 = block_Synthetic( 4, 1, (complex_t) 2.0);
  block_t *W2 = block_Synthetic( 4, 1, (complex_t) 3.0);

  integer_t  p = 3;
  integer_t  k[4] = {1, 1, 1, 1};
  integer_t  n[4] = {4, 4, 4, 4};

  matrix_t* R = matrix_Reduced(p, n, k);



	block_Deallocate( V0 );
	block_Deallocate( W1 );
	block_Deallocate( V1 );
	block_Deallocate( W2 );
  matrix_Deallocate( R );

	fprintf(stderr, "\nTest result: PASSED.\n");

	fprintf(stderr, "\n");
	return 0;
}
