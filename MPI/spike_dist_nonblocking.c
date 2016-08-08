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

#include "spike_dist_interfaces.h"

Error_t spike_dist_nonblocking( matrix_t *A, block_t *x, block_t *f, const integer_t nrhs)
{
	//MPI initialize
	//int provided;
	//MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	//printf("Provided: %d funneled %d\n", provided, MPI_THREAD_MULTIPLE);
	//MPI_Init (&argc, &argv);
	int rank, size, master = 0;
	MPI_Status  status;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);


	if(rank == master) CheckPreprocessorMacros();

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	spike_timer_t start_t, end_t;
	Error_t error;

	if(rank == master){

		fprintf(stderr, "\nShared Memory Spike Solver.\n");
		start_t = GetReferenceTime();

		/* compute an optimal solving strategy */
		sm_schedule_t* S = spike_solve_analysis( A, nrhs, size );

		/* create the reduced sytem in advanced, based on the solving strategy */
		matrix_t* R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
		block_t*  xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );

		/* -------------------------------------------------------------------- */
		/* .. Factorization Phase. */
		/* -------------------------------------------------------------------- */
		integer_t p;
		for(p=1; p < S->p; p++)
		{
			sendSchedulePacked(S, p);
			const integer_t r0 = S->n[p];
			const integer_t rf = S->n[p+1];

			matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf, _DIAG_BLOCK_);
			sendMatrixPacked(Aij, p, AIJ_TAG);

			block_t*  fi  = block_ExtractBlock( f, r0, rf );
			block_SetBandwidthValues( fi, S->ku[p], S->kl[p] );

			sendBlockPacked(fi, p, FI_TAG);
			
			if(p == 0){

				matrix_t* Bi = matrix_ExtractMatrix (A, r0, rf, rf, rf + A->ku, _B_BLOCK_);

				sendMatrixPacked(Bi, p, BI_TAG);
				matrix_Deallocate( Bi );
			}

			else if (p == ( S->p -1)){

				matrix_t* Ci = matrix_ExtractMatrix (A, r0, rf, r0 - A->kl, r0, _C_BLOCK_);
				sendMatrixPacked(Ci, p, CI_TAG);
				matrix_Deallocate( Ci );
			}

			else{

				matrix_t* Bi = matrix_ExtractMatrix(A, r0, rf, rf, rf + A->ku, _B_BLOCK_);
				matrix_t* Ci = matrix_ExtractMatrix(A, r0, rf, r0 - A->kl, r0 ,_C_BLOCK_);
				matrix_PrintAsDense(Bi, "Bi Sended");
				sendMatrixPacked(Bi, p, BI_TAG);				
				sendMatrixPacked(Ci, p, CI_TAG);

				matrix_Deallocate( Bi );
				matrix_Deallocate( Ci );
			}

			matrix_Deallocate( Aij);
			block_Deallocate (fi );
		}

		//MASTER WORK
		/* Set up solver handler */
		DirectSolverHander_t *handler = directSolver_CreateHandler();
		
		directSolver_Configure(handler);

		p = 0;
		const integer_t r0 = S->n[p];
		const integer_t rf = S->n[p+1];
		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf, _DIAG_BLOCK_);

		directSolver_Factorize( handler,
			Aij->n,
			Aij->nnz,
			Aij->colind,
			Aij->rowptr,
			Aij->aij);

		/* solve the system for the RHS value */
		block_t*  fi = block_ExtractBlock( f, r0, rf );
		block_t*  yi = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
		block_SetBandwidthValues( yi, S->ku[p], S->kl[p] );
		block_SetBandwidthValues( fi, S->ku[p], S->kl[p] );

		/* Solve Ai * yi = fi */
		directSolver_SolveForRHS( handler, nrhs, yi->aij, fi->aij);

		/* Extract the tips of the yi block */
		block_t* yit = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
		block_t* yib = block_ExtractTip( yi, _BOTTOM_SECTION_, _COLMAJOR_ );
		block_AddTipTOReducedRHS(rank, S->ku, S->kl, xr, yit);
		block_AddTipTOReducedRHS(rank, S->ku, S->kl, xr, yib);

		/* clean up */
		block_Deallocate (yi );
		block_Deallocate (yit);
		block_Deallocate (yib);

		
		block_t* Vi = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
		block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

		/* solve Ai * Vi = Bi */
		directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij);

		block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
		block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
		matrix_AddTipToReducedMatrix( S->p, rank, S->n, S->ku, S->kl, R, Vit);
		matrix_AddTipToReducedMatrix( S->p, rank, S->n, S->ku, S->kl, R, Vib);

		block_t* Bib = block_ExtractTip( Bi, _BOTTOM_SECTION_, _COLMAJOR_ );

		block_Deallocate( Bi );
		block_Deallocate( Vi);
		block_Deallocate( Vit);
		block_Deallocate( Vib);
		block_Deallocate( fi);
		//matrix_Deallocate( Aij);

		integer_t i;
		for(i=0; i<6*(size-1)-2; i++){
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			printf("MASTER MPI_STATUS.TAG: %d, rank: %d\n", status.MPI_TAG, status.MPI_SOURCE);
			switch(status.MPI_TAG) {
				case VIWI_TAG:
				{
					block_t* b = recvBlockPacked(status.MPI_SOURCE, VIWI_TAG);
					matrix_AddTipToReducedMatrix( S->p, status.MPI_SOURCE, S->n, S->ku, S->kl, R, b);
					block_Deallocate(b);
					break;
				}
				case YI_TAG:
				{
					/* Add the tips of the yi block to the reduced RHS */
					block_t* b = recvBlockPacked(status.MPI_SOURCE, YI_TAG);
					block_AddTipTOReducedRHS(status.MPI_SOURCE, S->ku, S->kl, xr, b);
					block_Deallocate(b);
					break;
				}
			}
		}
		printf("Done\n");
		MPI_Barrier(MPI_COMM_WORLD);
		/* -------------------------------------------------------------------- */
		/* .. Solution of the reduced system.                                   */
		/* -------------------------------------------------------------------- */

		block_t* yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
		fprintf(stderr, "\nSolving reduced linear system\n");
		directSolver_Host_Solve ( R->n, R->nnz, xr->m, R->colind, R->rowptr, R->aij, yr->aij, xr->aij);
		matrix_PrintAsDense(R, "Reduced System");
		block_Print(xr, "X Reduced System");
		block_Print(yr, "yr Reduced System");


		/* Free some memory, yr and R are not needed anymore */
		block_Deallocate ( xr );
		matrix_Deallocate( R  );

		/* -------------------------------------------------------------------- */
		/* .. Backward substitution phase. */
		/* -------------------------------------------------------------------- */
		for(p=1; p < S->p; p++)
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
			sendBlockPacked(xi, p, XI_TAG);

			/* extract fi sub-block */
			block_t*  fi  = block_ExtractBlock(f, obs, obe );
			sendBlockPacked(fi, p, FI_TAG);
			
			if ( p == 0 ){

				block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p]);
				sendBlockPacked(xt_next, p,XT_NEXT_TAG);
				block_Deallocate (xt_next);
			}

			else if ( p == ( S->p -1)){

				block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );
				sendBlockPacked(xb_prev, p, XT_PREV_TAG);
				block_Deallocate (xb_prev);
			}

			else{
				block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p]);
				sendBlockPacked(xt_next, p, XT_NEXT_TAG);
				block_Deallocate (xt_next);
				
				block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );
				sendBlockPacked(xb_prev, p, XT_PREV_TAG);
				block_Deallocate (xb_prev);
			}
			block_Deallocate( fi );
			block_Deallocate( xi );
		}
		

		//MASTER WORK
		p = 0; 
		const integer_t obs = S->n[p];        		/* original system starting row */
		const integer_t obe = S->n[p+1];	  		/* original system ending row   */
		const integer_t rbs = S->r[p];		  		/* reduceed system starting row */
		const integer_t rbe = S->r[p+1];			/* reduced system ending row    */
		const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */

		block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p]);
		block_t*  xi  = block_ExtractBlock(x, obs, obe );
		block_t*  fi2  = block_ExtractBlock(f, obs, obe );

		block_Print(Bib, "Bib Block");
		block_Print(xt_next, "xt_next Block");

		/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
		gemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
			Bib->n,    						/* m - number of rows of A    */
			xt_next->m, 					/* n - number of columns of B */
			Bib->m,    						/* k - number of columns of A */
			__nunit,						/* alpha                      */
			Bib->aij, 						/* A block                    */
			Bib->n,    						/* lda - first dimension of A */
			xt_next->aij, 					/* B block                    */
			xt_next->n,    					/* ldb - first dimension of B */
			__punit,						/* beta                       */
			&fi2->aij[ni - S->ku[p]], 		/* C block                    */
			ni ); 					 		/* ldc - first dimension of C */

		block_Print(fi2, "Fi Block");
		block_Print(xi, "xi Block");

		/* Solve Ai * xi = fi */
		directSolver_SolveForRHS(handler, xi->m, xi->aij, fi2->aij);
		// directSolver_Host_Solve ( Aij->n, Aij->nnz, fi2->m, Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi2->aij);
		block_Print(xi, "xi Block");
		block_AddBlockToRHS(x, xi, S->n[rank], S->n[rank+1]);
	

		directSolver_Finalize(handler);
		block_Deallocate (fi2 );
		block_Deallocate ( Bib );
		block_Deallocate ( xt_next);

		block_AddBlockToRHS(x, xi, S->n[rank], S->n[rank+1]);

		for(p=1; p < S->p; p++){
			MPI_Probe(MPI_ANY_SOURCE, XI_TAG, MPI_COMM_WORLD, &status);
			block_t* xi = recvBlockPacked(status.MPI_SOURCE, XI_TAG);
			block_AddBlockToRHS(x, xi, S->n[status.MPI_SOURCE], S->n[status.MPI_SOURCE+1]);
			block_Deallocate ( xi );
		}

		block_Deallocate( xi );
		schedule_Destroy( S );
		block_Deallocate( yr);
	
		end_t = GetReferenceTime();

		fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);
		block_Print( x, "Solution of the linear system");

		ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
	
		// fprintf(stderr, "\nPARDISO REFERENCE SOLUTION...\n");
		// SolveOriginalSystem( A, x, f);

		fprintf(stderr, "\nProgram finished\n");
	}

	else{ //WORKERS

		//Initializing values for workers
		integer_t p = rank;
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
			Aij->aij);
		
		if(rank == 0 || rank == size-1) max_work = 2;
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
					matrix_t* BiTmp = recvMatrixPacked(master, BI_TAG);
					matrix_PrintAsDense(BiTmp, "Bi Recived");
					block_t* Bi = block_BuildBlockFromMatrix(BiTmp, _V_BLOCK_, BiTmp->n, BiTmp->n, S->ku[p], S->kl[p]);
					block_Print(Bi, "Bi");
					//matrix_Deallocate( BiTmp );

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
					matrix_t* CiTmp = recvMatrixPacked(master, CI_TAG);
					block_t* Ci = block_BuildBlockFromMatrix(CiTmp, _W_BLOCK_, CiTmp->n, CiTmp->n, S->ku[p], S->kl[p]);
					block_Print(Ci, "Ci");
					//matrix_Deallocate( CiTmp );

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
		printf("Done\n");
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

		if(rank == 0 || rank == size-1) max_work = 1;
		else max_work = 2;

		for(i=0; i<max_work; i++){
			MPI_Probe(master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			switch(status.MPI_TAG) {
				case XT_NEXT_TAG:
				{
					block_t* xt_next = recvBlockPacked(master, XT_NEXT_TAG);
			
					/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
					gemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
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
					gemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
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
		
	}
	debug("Number of malloc() calls %d, number of free() calls %d\n", cnt_alloc, cnt_free );
	debug("Rank %d Finished!\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	return 0;
}
