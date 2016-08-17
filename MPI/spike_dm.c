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

#include "spike_dm.h"

Error_t spike_dm( matrix_t *A, block_t *x, block_t *f, const integer_t nrhs)
{

	int rank, size, master = 0;
	MPI_Status  status;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);


	if(rank == master) CheckPreprocessorMacros();

	//Local variables.
	spike_timer_t start_t, end_t;
	Error_t error;

	if(rank == master){


		fprintf(stderr, "\nDistributed Memory Spike Solver.\n");
		start_t = GetReferenceTime();

		/* compute an optimal solving strategy */
		debug("Solving analisis and generating schedule");
		sm_schedule_t* S = spike_solve_analysis( A, nrhs );

		/* create the reduced sytem in advanced, based on the solving strategy */
		debug("Creating Reduced System");
		matrix_t* R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
		block_t*  xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );

		/* Set up solver handler */
		DirectSolverHander_t *handler = directSolver_CreateHandler();
		directSolver_Configure(handler, S->max_nrhs );

		/* Send structures to all nodes */
		debug("Sending schedule to all nodes");
		scatterSchedule(S);
		debug("Sending Aij, Bi, Ci, Fi to all nodes");
		scatterAijBiCiFi(S, A, f);

		integer_t p = rank;

		//Initialize Bib, we will need it later on the Backward phase.
		block_t* Bib = block_CreateEmptyBlock( S->kl[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _TOP_SECTION_ );

		//Master Factorization and Solve Aij*fi and Aij*Bi.
		if(MASTER_WORKING){
			debug("Master WORKING NOW - Factorize");
			masterWorkFactorize(handler, S, A, f, R, Bib, xr, nrhs);
		}
		
		//Reciving blocks from all nodes and adding to R and xr.
		debug("Reciving Reduced System");
		gatherReducedSystem(S, R, xr);

		//Sincronization of all nodes.
		MPI_Barrier(MPI_COMM_WORLD);

		/* -------------------------------------------------------------------- */
		/* .. Solution of the reduced system.                                   */
		/* -------------------------------------------------------------------- */
		debug("Solving Reduced System");
		block_t* yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

		fprintf(stderr, "\nSolving reduced linear system\n");
		directSolver_Host_Solve ( R->n, R->nnz, xr->m, R->colind, R->rowptr, R->aij, yr->aij, xr->aij);

		matrix_PrintAsDense(R, "Reduced System");
		block_Print(xr, "X Reduced System");
		block_Print(yr, "yr Reduced System");


		/* Free some memory, xr and R are not needed anymore */
		block_Deallocate ( xr );
		matrix_Deallocate( R  );

		//Send x, f and tips: xt_next and xb_prev
		debug("Sending solution of reduced system to all nodes");
		scatterXiFi(S, x, f, yr);

		//Solving Backward Solution.
		if(MASTER_WORKING){
			debug("Master WORKING NOW - Backward");
			masterWorkBackward(S, yr, f, x, Bib, handler);
		}
		block_Deallocate(Bib);

		//Recive all xi and add it to final solution x
		debug("Reciving final solution");
		gatherXi(S, x);

		//Cleaning up
		directSolver_Finalize(handler);
		schedule_Destroy( S );
		block_Deallocate( yr);
	
		end_t = GetReferenceTime();

		fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);
		block_Print( x, "Solution of the linear system");

		ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);

		fprintf(stderr, "\nProgram finished\n");
	}

	else{ //WORKERS

		//Initializing values for workers
		integer_t p;
		if(MASTER_WORKING){
			p = rank;
		}
		else
		{	
			p = rank -1;
		}
		debug("Reciving schedule");
		sm_schedule_t* S = recvSchedulePacked(master);
		block_t* Bib = block_CreateEmptyBlock( S->kl[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _TOP_SECTION_ );
		block_t* Cit = block_CreateEmptyBlock( S->kl[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _TOP_SECTION_ );

		//Set up solver handler
		DirectSolverHander_t *handler = directSolver_CreateHandler();
		directSolver_Configure(handler, S->max_nrhs );

		//Wait for AIJ and factorize matrix
		debug("Reciving Aij and starting to factorize");
		matrix_t* Aij = recvMatrixPacked(master, AIJ_TAG);
		directSolver_Factorize( handler, Aij->n, Aij->nnz, Aij->colind, Aij->rowptr, Aij->aij);
		
		//Solve Aij*Bi, Aij*Ci and Aij*fi
		debug("Solving AIJ Aij*Bi, Aij*Ci and Aij*fi");
		workerSolveAndSendTips(S, master, nrhs, Aij, Bib, Cit, handler);

		//Master Solving Reduced System.
	
		//Syncronize all process
		MPI_Barrier(MPI_COMM_WORLD);
		
		//Solving Backward Solution.
		debug("Solving Backward solution");
		workerSolveBackward(S, Bib, Cit, master, handler);

		/* Show statistics and clean up solver internal memory */
		directSolver_ShowStatistics(handler);
		directSolver_Finalize(handler);

		matrix_Deallocate(Aij);	
		schedule_Destroy( S );
		
	}
	debug("Number of malloc() calls %d, number of free() calls %d\n", cnt_alloc, cnt_free );
	debug("Rank %d Finished!\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	return 0;
}
