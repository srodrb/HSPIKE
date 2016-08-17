/* *
 * =====================================================================================
 *
 *       Filename:  split.c
 *
 *    Description:  Splits the original matrix into two sub-matrices.
 *
 *        Version:  1.0
 *        Created:  05/07/16 09:59:35
 *       Revision:  Checked for correctness and memory leaks.
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 *
 * [ 1.917  0.766  0.     0.     0.     0.     0.     0.     0.     0.   ]
 * [ 0.581  1.589  0.262  0.     0.     0.     0.     0.     0.     0.   ]
 * [ 0.     0.677  1.265  0.122  0.     0.     0.     0.     0.     0.   ]
 * [ 0.     0.     0.687  1.783  0.386  0.     0.     0.     0.     0.   ]
 * [ 0.     0.     0.     0.439  1.918  0.84   0.     0.     0.     0.   ]
 * [ 0.     0.     0.     0.     0.321  1.827  0.278  0.     0.     0.   ]
 * [ 0.     0.     0.     0.     0.     0.571  1.728  0.07   0.     0.   ]
 * [ 0.     0.     0.     0.     0.     0.     0.48   1.26   0.633  0.   ]
 * [ 0.     0.     0.     0.     0.     0.     0.     0.861  1.912  0.585]
 * [ 0.     0.     0.     0.     0.     0.     0.     0.     0.835  1.261]
 *
 *
 * =====================================================================================
 */

#include "spike_mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
	integer_t rank, size, master=0;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	Error_t  res = 0;
	if(rank == master){
		fprintf(stderr, "\nSPLIT TEST.\n\tINFO:Divides the input matrix into two sub-blocks, checking for correctness");

		matrix_t* A = matrix_LoadCSR("../Tests/split/matrix.bin");
		printf("Load Matrix... Done\n");

		// extracts the second block, loads the reference and compares them
		matrix_t* Block1 = matrix_ExtractMatrix(A, 0,  5, 0,  5);
		matrix_t* Block2 = matrix_ExtractMatrix(A, 5, 10, 5, 10);
		printf("Divide Matrix... Done\n");
		sendAij (Block1, 1);
		sendAij (Block2, 2);
		printf("Send Matrix... Done\n");
	 
	}

	else if (rank == 1){
		matrix_t* Block1 = recvAij(master);
		matrix_t* Block1_ref = matrix_LoadCSR("../Tests/split/block1.bin");
		res = matrix_AreEqual( Block1_ref, Block1 );

		if ( res == 0 )
		{
			fprintf(stderr, "Test FAILED for first block, matrices are not equal\n");
			exit(-1);
		}
		matrix_Print(Block1_ref, "First block (reference)");
		matrix_Print(Block1    , "First block (created)");

	}
	
	else if (rank == 2){
		matrix_t* Block2 = recvAij(master);
		matrix_t* Block2_ref = matrix_LoadCSR("../Tests/split/block2.bin");
		res = matrix_AreEqual( Block2_ref, Block2 );

		if ( res == 0 )
		{
			fprintf(stderr, "Test FAILED for second block, matrices are not equal\n");
			exit(-1);
		}
		matrix_Print(Block2_ref, "Second block (reference)");
		matrix_Print(Block2    , "Second block (created)");
	
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	// clean up
	//matrix_Deallocate( A );
	//matrix_Deallocate( Block1_ref );
	//matrix_Deallocate( Block2_ref );
	//matrix_Deallocate( Block1 );
	//matrix_Deallocate( Block2 );

	if(rank == master)fprintf(stderr, "\nTest result: PASSED.\n");

	//matrix_t* B = matrix_LoadCSR("Tests/bandwidth/banded.bin");
	//compute_bandwidth( B );
	//matrix_Deallocate( B );

	
	//fprintf(stderr, "\n");
	MPI_Finalize();
	return 0;
}
