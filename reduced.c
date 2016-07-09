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
static block_t* block_Synthetic( const integer_t n, const integer_t m, const complex_t value, blocktype_t type )
{
  block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));

	B->type = type;
  B->n = n;
  B->m = m;
  B->aij = (complex_t*) spike_malloc( ALIGN_COMPLEX, n * m, sizeof(complex_t));

  for (integer_t i = 0; i < n * m; i++)
    B->aij[i] = value;

  return (B);
}


int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nREDUCED SYSTEM ASSEMBLY TEST.\n\tINFO:"
                  "Creates a reduced system from the synthetic V and W blocks.");

	Error_t  res = 0;

  /* Create some synthetic spikes Vi, Wi */
  block_t *V0 = block_Synthetic( 4, 2, (complex_t) 2.2, (blocktype_t) _V_BLOCK_ );
  block_t *W1 = block_Synthetic( 4, 1, (complex_t) 3.3, (blocktype_t) _W_BLOCK_ );
  block_t *V1 = block_Synthetic( 4, 1, (complex_t) 4.4, (blocktype_t) _V_BLOCK_ );
  block_t *W2 = block_Synthetic( 4, 1, (complex_t) 5.5, (blocktype_t) _W_BLOCK_ );

  integer_t  p     = 3;
  integer_t  ku[3] = {2, 1, 1};
  integer_t  kl[3] = {2, 1, 1};
  integer_t  n [4] = {0, 4, 8, 12};

  matrix_t* R = matrix_CreateEmptyReduced(p, n, ku, kl);

  res = matrix_FillReduced(p, 0, n, ku, kl, R, V0 );
  res = matrix_FillReduced(p, 1, n, ku, kl, R, W1 );
  res = matrix_FillReduced(p, 1, n, ku, kl, R, V1 );
  res = matrix_FillReduced(p, 2, n, ku, kl, R, W2 );

	block_Deallocate( V0 );
	block_Deallocate( W1 );
	block_Deallocate( V1 );
	block_Deallocate( W2 );
  matrix_Deallocate( R );

	fprintf(stderr, "\nTest result: PASSED.\n");

	fprintf(stderr, "\n");
	return 0;
}
