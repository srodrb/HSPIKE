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

#include "spike_analysis.h"
#include "spike_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Error_t SolveOriginalSystem( matrix_t *A, block_t *x, block_t *rhs )
{
	// local variables
	double start_t, end_t;
	Error_t error;

	fprintf(stderr, "\nSolving original linear system using reference direct solver");

	start_t = GetReferenceTime();
	error = system_solve( A->colind, A->rowptr, A->aij, x->aij, rhs->aij, A->n, rhs->m );
	end_t = GetReferenceTime();

	fprintf(stderr, "\nReference direct solver took %.6lf seconds", end_t - start_t );

	return (SPIKE_SUCCESS);
};

int main(int argc, char *argv[])
{
	//MPI initialize
	MPI_Init (&argc, &argv);	
	int rank, size, master = 0;
	MPI_Status  status;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);


	if(rank == master) CheckPreprocessorMacros();

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	timer_t start_t, end_t;
	const integer_t nrhs = 1;
	Error_t error;

	if(rank == master){

		fprintf(stderr, "\nShared Memory Spike Solver.\n");


		/* -------------------------------------------------------------------- */
		/* .. Load and initalize the system Ax=f. */
		/* -------------------------------------------------------------------- */
		// matrix_t* A = matrix_LoadCSR("../Tests/spike/penta_15.bin");
		matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large.bin");
		// matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");
		// matrix_PrintAsDense( A, "Original coeffient matrix" );

		// Compute matrix bandwidth
		block_t*  x = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
		block_t*  f = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

		block_InitializeToValue( x, __zero  ); // solution of the system
		block_InitializeToValue( f, __punit ); // rhs of the system

		start_t = GetReferenceTime();

		/* compute an optimal solving strategy */
		sm_schedule_t* S = spike_solve_analysis( A, nrhs, size-1 );

		/* create the reduced sytem in advanced, based on the solving strategy */
		matrix_t* R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
		block_t*  xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );

		/* -------------------------------------------------------------------- */
		/* .. Factorization Phase. */
		/* -------------------------------------------------------------------- */
		
		for(integer_t p=0; p < S->p; p++)
		{
			sendSchedulePacked(S, p+1);
			const integer_t r0 = S->n[p];
			const integer_t rf = S->n[p+1];

			matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
			sendMatrixPacked(Aij, p+1, AIJ_TAG);

			block_t*  fi  = block_ExtractBlock( f, r0, rf );
			block_SetBandwidthValues( fi, S->ku[p], S->kl[p] );

			sendBlockPacked(fi, p+1, FI_TAG);
			
			if(p == 0){

				block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );
				sendBlockPacked(Bi, p+1, BI_TAG);
				block_Deallocate( Bi );
			}

			else if (p == ( S->p -1)){

				block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );
				sendBlockPacked(Ci, p+1, CI_TAG);
				block_Deallocate( Ci );
			}

			else{

				block_t* Bi = matrix_ExtractBlock( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );
				block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

				sendBlockPacked(Bi, p+1, BI_TAG);				
				sendBlockPacked(Ci, p+1, CI_TAG);

				block_Deallocate( Bi );
				block_Deallocate( Ci );
			}

			matrix_Deallocate( Aij);
			block_Deallocate (fi );
		}

		integer_t i;
		for(i=0; i<6*(size-1)-4; i++){
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			//printf("MPI_STATUS.TAG: %d, rank: %d\n", status.MPI_TAG, rank);
			switch(status.MPI_TAG) {
				case VIWI_TAG:
				{
					block_t* b = recvBlockPacked(status.MPI_SOURCE, VIWI_TAG);
					matrix_AddTipToReducedMatrix( S->p, status.MPI_SOURCE-1, S->n, S->ku, S->kl, R, b);
					break;
				}
				case YI_TAG:
				{
					/* Add the tips of the yi block to the reduced RHS */
					block_t* b = recvBlockPacked(status.MPI_SOURCE, YI_TAG);
					block_AddTipTOReducedRHS(status.MPI_SOURCE-1, S->ku, S->kl, xr, b);
					break;
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		/* -------------------------------------------------------------------- */
		/* .. Solution of the reduced system.                                   */
		/* -------------------------------------------------------------------- */

		block_t* yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
		fprintf(stderr, "\nSolving reduced linear system\n");
		system_solve ( R->colind, R->rowptr, R->aij, yr->aij, xr->aij, R->n, xr->m);
		//block_Print(yr, "Solucion del sistema reducido");


		/* Free some memory, yr and R are not needed anymore */
		block_Deallocate ( xr );
		matrix_Deallocate( R  );

		/* -------------------------------------------------------------------- */
		/* .. Backward substitution phase. */
		/* -------------------------------------------------------------------- */
		for(integer_t p=0; p < S->p; p++)
		{
			fprintf(stderr, "Processing backward solution for the %d-th block\n", p);

			/* compute the limits of the blocks */
			const integer_t obs = S->n[p];        		/* original system starting row */
			const integer_t obe = S->n[p+1];	  		/* original system ending row   */
			const integer_t rbs = S->r[p];		  		/* reduceed system starting row */
			const integer_t rbe = S->r[p+1];			/* reduced system ending row    */
			const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */

			/* allocate pardiso configuration parameters */
			MKL_INT pardiso_conf[64];

			/* extract xi sub-block */
			block_t*  xi  = block_ExtractBlock(x, obs, obe );
			sendBlockPacked(xi, p+1, XI_TAG);

			/* extract fi sub-block */
			block_t*  fi  = block_ExtractBlock(f, obs, obe );
			sendBlockPacked(fi, p+1, FI_TAG);
			
			if ( p == 0 ){

				block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
				sendBlockPacked(xt_next, p+1,XT_NEXT_TAG);
				block_Deallocate (xt_next);
			}

			else if ( p == ( S->p -1)){

				block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );
				sendBlockPacked(xb_prev, p+1, XT_PREV_TAG);
				block_Deallocate (xb_prev);
			}

			else{
				block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
				sendBlockPacked(xt_next, p+1, XT_NEXT_TAG);
				block_Deallocate (xt_next);
				
				block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );
				sendBlockPacked(xb_prev, p+1, XT_PREV_TAG);
				block_Deallocate (xb_prev);
			}
			MPI_Probe(MPI_ANY_SOURCE, XI_TAG, MPI_COMM_WORLD, &status);
			xi = recvBlockPacked(status.MPI_SOURCE, XI_TAG);
			block_AddBlockToRHS(x, xi, S->n[status.MPI_SOURCE-1], S->n[status.MPI_SOURCE]);
			block_Deallocate    ( xi );
			block_Deallocate 	( fi );
		}
		schedule_Destroy( S );
		block_Deallocate( yr);
	
		end_t = GetReferenceTime();

		fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);
		block_Print( x, "Solution of the linear system");

		ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
	
		// fprintf(stderr, "\nPARDISO REFERENCE SOLUTION...\n");
		// SolveOriginalSystem( A, x, f);

		/* -------------------------------------------------------------------- */
		/* .. Clean up. */
		/* -------------------------------------------------------------------- */
		matrix_Deallocate ( A );
		block_Deallocate  ( x );
		block_Deallocate  ( f );

		/* -------------------------------------------------------------------- */
		/* .. Load and initalize the system Ax=f. */
		/* -------------------------------------------------------------------- */
		fprintf(stderr, "\nProgram finished\n");
	}

	else{ //WORKERS

		//Initializing values for workers
		integer_t p = rank -1;
		sm_schedule_t* S = recvSchedulePacked(master);
		const integer_t r0 = S->n[p];
		const integer_t rf = S->n[p+1];
		MKL_INT pardiso_conf[64];
		block_t* Bib;
		block_t* Cit;
		integer_t max_work;

		/* Set up solver handler */
		DirectSolverHander_t *handler = directSolver_CreateHandler();
		
		directSolver_Configure(handler);

		/*Wait for AIJ and factorize matrix */
		matrix_t* Aij = recvMatrixPacked(master, AIJ_TAG);

		directSolver_Factorize( handler,
			Aij->n,
			Aij->nnz,
			Aij->colind,
			Aij->rowptr,
			Aij->aij, Aij->n);
		
		if(rank == 1 || rank == size-1) max_work = 2;
		else max_work = 3;
		integer_t i;

		for(i=0; i<max_work; i+=1){
			MPI_Probe(master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			//printf("MPI_STATUS.TAG: %d, rank: %d\n", status.MPI_TAG, rank);
			switch(status.MPI_TAG) {
				case FI_TAG:
				{
					/* solve the system for the RHS value */
					block_t*  fi = recvBlockPacked(master, FI_TAG);
					block_t*  yi = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
					block_SetBandwidthValues( yi, S->ku[p], S->kl[p] );
					//block_t*  yi = recvBlock(master);

					/* Solve Ai * yi = fi */
					directSolver_SolveForRHS( handler, nrhs, yi->aij, fi->aij);


					/* Extract the tips of the yi block */
					block_t* yit = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
					block_t* yib = block_ExtractTip( yi, _BOTTOM_SECTION_, _COLMAJOR_ );
					sendBlockPacked(yit, master, YI_TAG);
					sendBlockPacked(yib, master, YI_TAG);

					/* clean up */
					block_Deallocate (fi );
					block_Deallocate (yi );
					block_Deallocate (yit);
					block_Deallocate (yib);
					
					break;
				}

				case BI_TAG:
				{
					block_t* Vi = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
					block_t* Bi = recvBlockPacked(master, BI_TAG);

					/* solve Ai * Vi = Bi */
					directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij);

					block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
					block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
					sendBlockPacked(Vit, master, VIWI_TAG);
					sendBlockPacked(Vib, master, VIWI_TAG);

					Bib = block_ExtractTip( Bi, _BOTTOM_SECTION_, _COLMAJOR_ );

					block_Deallocate( Bi );
					block_Deallocate( Vi);
					block_Deallocate( Vit);
					block_Deallocate( Vib);

					break;
				}

				case CI_TAG:
				{
					block_t* Wi = block_CreateEmptyBlock( rf - r0, S->kl[p], S->ku[p], S->kl[p], _W_BLOCK_, _WHOLE_SECTION_ );
					block_t* Ci = recvBlockPacked(master, CI_TAG);

					/* solve Ai  * Wi = Ci */
					directSolver_SolveForRHS( handler, Wi->m, Wi->aij, Ci->aij);

					block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
					block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
					sendBlockPacked(Wit, master, VIWI_TAG);
					sendBlockPacked(Wib, master, VIWI_TAG);
			
					Cit = block_ExtractTip(Ci, _TOP_SECTION_, _COLMAJOR_ );			
		
					block_Deallocate( Ci );
					block_Deallocate( Wi );
					block_Deallocate( Wit);
					block_Deallocate( Wib);
				
					break;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		/* -------------------------------------------------------------------- */
		/* .. Solution of the reduced system.                                   */
		/* -------------------------------------------------------------------- */

		/* compute the limits of the blocks */
		const integer_t obs = S->n[p];        		/* original system starting row */
		const integer_t obe = S->n[p+1];	  		/* original system ending row   */
		const integer_t rbs = S->r[p];		  		/* reduceed system starting row */
		const integer_t rbe = S->r[p+1];			/* reduced system ending row    */
		const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */
			
		block_t* xi = recvBlockPacked(master, XI_TAG);
		block_t* fi = recvBlockPacked(master, FI_TAG);

		if(rank == 1 || rank == size-1) max_work = 1;
		else max_work = 2;

		for(i=0; i<max_work; i++){
			MPI_Probe(master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			switch(status.MPI_TAG) {
				case XT_NEXT_TAG:
				{
					block_t* xt_next = recvBlockPacked(master, XT_NEXT_TAG);
			
					/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
					cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
						Bib->n,    						/* m - number of rows of A    */
						xt_next->m, 					/* n - number of columns of B */
						Bib->m,    						/* k - number of columns of A */
						__nunit,						/* alpha                      */
						Bib->aij, 						/* A block                    */
						Bib->n,    						/* lda - first dimension of A */
						xt_next->aij, 					/* B block                    */
						xt_next->n,    					/* ldb - first dimension of B */
						__punit,						/* beta                       */
						&fi->aij[ni - S->ku[p]], 		/* C block                    */
						ni ); 					 		/* ldc - first dimension of C */
			
					/* Solve Ai * xi = fi */
					directSolver_SolveForRHS(handler, xi->m, xi->aij, fi->aij);

					block_Deallocate ( Bib );
					block_Deallocate ( xt_next);

					break;
				}
				case XT_PREV_TAG:
				{
					block_t* xb_prev = recvBlockPacked(master, XT_PREV_TAG);
					/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */ 
					cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
						Cit->n,    						/* m - number of rows of A    */
						xb_prev->m, 					/* n - number of columns of B */
						Cit->m,    						/* k - number of columns of A */
						__nunit,						/* alpha                      */
						Cit->aij, 						/* A block                    */
						Cit->n,    						/* lda - first dimension of A */
						xb_prev->aij, 					/* B block                    */
						xb_prev->n,    					/* ldb - first dimension of B */
						__punit,						/* beta                       */
						fi->aij, 			 		    /* C block                    */
						ni );		 					/* ldc - first dimension of C */

					/* Solve Ai * xi = fi */
					directSolver_SolveForRHS(handler, xi->m, xi->aij, fi->aij);

					block_Deallocate ( Cit );
					block_Deallocate ( xb_prev);
					break;
				}
			}
		}
		sendBlockPacked(xi, master, XI_TAG);

		/* Show statistics and clean up solver internal memory */
		directSolver_ShowStatistics(handler);
		directSolver_Finalize(handler);

		block_Deallocate ( xi );
		block_Deallocate ( fi );
		matrix_Deallocate(Aij);	
		schedule_Destroy  ( S );

		debug("Number of malloc() calls %d, number of free() calls %d\n", cnt_alloc, cnt_free );
		
	}
	debug("Rank %d Finished!\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
