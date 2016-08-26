/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  SPIKE usage demonstration
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:32:39
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#include "spike_analysis_dm.h"
#include "spike_datatypes.h"
#include "spike_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> 
		

int main(int argc, char *argv[])
{

	MPI_Init (&argc, &argv);	
	int rank, size, master = 0;
	MPI_Status  status;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	double start_t, end_t;
	const integer_t nrhs = 1;
	Error_t error;
	char msg[200];

	//matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");
	//matrix_t* A = matrix_LoadCSR("../Tests/heptadiagonal/medium.bin");
	//matrix_t* A = matrix_LoadCSR("../Tests/spike/15e10Matrix.bin");
	//matrix_t* A = matrix_LoadCSR("../Tests/spike/permuted.bsit");
	//matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large.bin");
	matrix_t* A = matrix_LoadCSR("../Tests/complex16/penta_1k.z");

	integer_t  p     = size - 1;
	integer_t  ku[3] = {2, 1, 1};
	integer_t  kl[3] = {2, 1, 1};
	integer_t  n [4] = {0, 5, 10, 15};


	dm_schedule_t* schedule;
	printf("Testing Send / Recv Schedule:\n");
	if(rank == master){ //MASTER
		start_t = GetReferenceTime();
		schedule = spike_solve_analysis( A, nrhs); //Number of partitions
		if(schedule->p != size){
			printf("Partitons and number of process must be equal, to fix this set MASTER_WORKING to 1\n");
			return 0;
		}
		//matrix_t* R = matrix_CreateEmptyReducedSystem( schedule->p, schedule->n, schedule->ku, schedule->kl);
		integer_t r0,rf,c0,cf;
		
		for(integer_t p=0; p<schedule->p-1; p++){
			int i, check = 0;
			sendSchedulePacked(schedule, p+1);
			dm_schedule_t* sTest = recvSchedulePacked(p+1);
			check += schedule->p - sTest->p;
			check += schedule->max_m - sTest->max_m;
			check += schedule->max_n - sTest->max_n;
			for(i=0; i<schedule->p+1; i++) check += schedule->n[i] - sTest->n[i];
			for(i=0; i<schedule->p+1; i++) check += schedule->r[i] - sTest->r[i];
			for(i=0; i<schedule->p  ; i++) check += schedule->ku[i] - sTest->ku[i];
			for(i=0; i<schedule->p  ; i++) check += schedule->kl[i] - sTest->kl[i];

			if(check == 0) printf("TEST Send Schedule: \t\t%d PASSED\n", p+1);
			else printf("TEST Send Schedule %d : \t\t%d FAIL \n", check, p+1);
			schedule_Destroy( sTest );
						
		}
		get_maximum_av_host_memory();
		MPI_Barrier(MPI_COMM_WORLD);
		/*for(integer_t p=0; p<schedule->p; p++){

			r0 = schedule->n[p];
			rf = schedule->n[p+1];

			matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
			
			sendMatrix(Aij, p+1);
			matrix_t* test = recvMatrix(p+1);
		
			if(matrix_AreEqual (test, Aij))printf("TEST send Matrix: \t\t%d PASSED\n", p+1);
			matrix_Deallocate(Aij);

		}*/
		
		printf("Testing Send / Recv Matrix:\n");
		for(integer_t p=0; p<schedule->p-1; p++){

			r0 = schedule->n[0];
			rf = schedule->n[1];

			matrix_t* Aij2 = matrix_ExtractMatrix(A, r0, rf, r0, rf, _DIAG_BLOCK_);
			
			sendMatrixPacked(Aij2, p+1, 0);
			matrix_t* test2 = recvMatrixPacked(p+1, 0);
		
			if(matrix_AreEqual (test2, Aij2))printf("TEST send Matrix Packed: \t%d PASSED\n", p+1);
			get_maximum_av_host_memory();
			
			matrix_Deallocate(Aij2);
			matrix_Deallocate(test2);

		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		printf("Testing Send / Recv Block:\n");
		for(integer_t p=0; p<schedule->p-1; p++){
			int r0 = schedule->n[p];
			int rf = schedule->n[p+1];
			block_t *V0Test = matrix_ExtractBlock (A, rf - A->ku, rf, rf, rf + A->ku, _B_BLOCK_);
			//block_t *V0Test = block_Synthetic( 5, 2, ku[0], kl[0], 2.0, _V_BLOCK_, _WHOLE_SECTION_ );
			sendBlockPacked(V0Test,p+1, 0);
			block_t *V0 = recvBlockPacked(p+1,0);
			if(block_AreEqual (V0Test, V0))printf("TEST Send Block Packed: \t%d PASSED\n", p+1);
			block_Deallocate (V0Test);
			block_Deallocate (V0);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for(integer_t p=0; p<schedule->p-1; p++){
			int r0 = schedule->n[0];
			int rf = schedule->n[1];
			block_t*  yi = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
			block_SetBandwidthValues( yi, schedule->ku[p], schedule->kl[p] );

			block_t *V0Test = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
			debug("Sending block");
			sendBlockPacked(V0Test,p+1, 0);
			debug("Reciving block");
			block_t *V0 = recvBlockPacked(p+1,0);

			if(block_AreEqual (V0Test, V0))printf("TEST Send Block Packed TIP: \t%d PASSED\n", p+1);

			block_Deallocate (V0Test);
			block_Deallocate (V0);
		}
		schedule_Destroy( schedule);
		
	}
	else{
		//Schedule Testing
		dm_schedule_t* sTest = recvSchedulePacked(master);
		sendSchedulePacked(sTest, master);
		MPI_Barrier(MPI_COMM_WORLD);

		//Matrix Testing
		matrix_t* Aij = recvMatrixPacked(master, 0);
		sendMatrixPacked(Aij, master, 0);
		MPI_Barrier(MPI_COMM_WORLD);

		//Block Testing
		block_t *V0 = recvBlockPacked(master,0);
		sendBlockPacked(V0, master, 0);
		MPI_Barrier(MPI_COMM_WORLD);

		V0 = recvBlockPacked(master,0);
		sendBlockPacked(V0, master, 0);
		block_Deallocate (V0);

	}
	MPI_Finalize();
	return 0;
}
