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
static block_t* block_Synthetic(const integer_t n,
                                const integer_t m,
                                const integer_t ku,
                                const integer_t kl, 
                                const complex_t value, 
                                blocktype_t type,
                                blocksection_t section )
{
  block_t* B = (block_t*) spike_malloc( ALIGN_INT, 1, sizeof(block_t));

	B->type = type;
  B->section = section;
  B->n    = n;
  B->m    = m;
  B->ku   = ku;
  B->kl   = kl;
  B->aij  = (complex_t*) spike_malloc( ALIGN_COMPLEX, n * m, sizeof(complex_t));

  for (integer_t row = 0; row < n; row++) {
    for(integer_t col = 0; col < m; col++ ) {
      B->aij[row + col * n] = (0.1 * (row +1)) + value;
    }
  }

  block_Print ( B, NULL );

  return (B);
};

int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nREDUCED SYSTEM ASSEMBLY TEST.\n\tINFO:"
                  "Creates a reduced system from the synthetic V and W blocks.");

  integer_t  p     = 3;
  integer_t  ku[3] = {2, 1, 1};
  integer_t  kl[3] = {2, 1, 1};
  integer_t  n [4] = {0, 5, 10, 15};

  /* Create some synthetic spikes Vi, Wi */
  block_t *V0 = block_Synthetic( 5, 2, ku[0], kl[0], 2.0, _V_BLOCK_, _WHOLE_SECTION_ );
  block_t *W1 = block_Synthetic( 5, 1, ku[1], kl[1], 3.0, _W_BLOCK_, _WHOLE_SECTION_ );
  block_t *V1 = block_Synthetic( 5, 1, ku[1], kl[1], 4.0, _V_BLOCK_, _WHOLE_SECTION_ );
  block_t *W2 = block_Synthetic( 5, 1, ku[2], kl[2], 5.0, _W_BLOCK_, _WHOLE_SECTION_ );

  matrix_t* R = matrix_CreateEmptyReducedSystem ( p, n, ku, kl );


  block_t *V0t = block_ExtractTip( V0, _TOP_SECTION_ );
  block_t *V0b = block_ExtractTip( V0, _BOTTOM_SECTION_ );
  
  block_t *W1t = block_ExtractTip( W1, _TOP_SECTION_ );
  block_t *W1b = block_ExtractTip( W1, _BOTTOM_SECTION_ );
  
  block_t *V1t = block_ExtractTip( V1, _TOP_SECTION_ );
  block_t *V1b = block_ExtractTip( V1, _BOTTOM_SECTION_ );
  
  block_t *W2t = block_ExtractTip( W2, _TOP_SECTION_ );
  block_t *W2b = block_ExtractTip( W2, _BOTTOM_SECTION_ );

  matrix_AddTipToReducedMatrix(p, 0, n, ku, kl, R, V0t );
  matrix_AddTipToReducedMatrix(p, 0, n, ku, kl, R, V0b );
  matrix_AddTipToReducedMatrix(p, 1, n, ku, kl, R, W1t );
  matrix_AddTipToReducedMatrix(p, 1, n, ku, kl, R, W1b );
  matrix_AddTipToReducedMatrix(p, 1, n, ku, kl, R, V1t );
  matrix_AddTipToReducedMatrix(p, 1, n, ku, kl, R, V1b );
  matrix_AddTipToReducedMatrix(p, 2, n, ku, kl, R, W2t );
  matrix_AddTipToReducedMatrix(p, 2, n, ku, kl, R, W2b );

  matrix_PrintAsDense( R, "Reduced system");

  /* Clean up */
  block_Deallocate(V0);
  block_Deallocate(V0t);
  block_Deallocate(V0b);
  
  block_Deallocate(W1);
  block_Deallocate(W1t);
  block_Deallocate(W1b);
  
  block_Deallocate(V1);
  block_Deallocate(V1t);
  block_Deallocate(V1b);
  
  block_Deallocate(W2);
  block_Deallocate(W2t);
  block_Deallocate(W2b);
  
  matrix_Deallocate(R);

	fprintf(stderr, "\nTest result: PASSED.\n");

	fprintf(stderr, "\n");
	return 0;
}
