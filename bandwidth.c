/*
 * =====================================================================================
 *
 *       Filename:  bandwidth.c
 *
 *    Description:  Test unit for bandwidth function
 *
 *        Version:  1.0
 *        Created:  05/07/16 14:14:47
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#include "spike_analysis.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nMATRIX BANDWIDTH TEST.\n\tINFO: Checks the correctness of the matrix envelope computation");

	/* ================================= */
	matrix_t* A = matrix_LoadCSR("Tests/bandwidth/banded.bin");

	compute_bandwidth( A );

	if ( A->ku != 2 )
	{
		fprintf(stderr, "TEST FAILED, upper bandwidth value is not correct\n");
		exit(-1);
	}

	if ( A->kl != 4 )
	{
		fprintf(stderr, "TEST FAILED, lower bandwidth value is not correct\n");
		exit(-1);
	}

	matrix_Deallocate( A );


	fprintf(stderr, "\nTest result: PASSED.\n");

	fprintf(stderr, "\n");
	return 0;
}
