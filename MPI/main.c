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
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);


	CheckPreprocessorMacros();

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	timer_t start_t, end_t;
	const integer_t nrhs = 1;
	Error_t error;
	char msg[200];

	if(rank == master){

		fprintf(stderr, "\nShared Memory Spike Solver.\n");


		/* -------------------------------------------------------------------- */
		/* .. Load and initalize the system Ax=f. */
		/* -------------------------------------------------------------------- */
		matrix_t* A = matrix_LoadCSR("../Tests/spike/penta_15.bin");
		//matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/large.bin");
		//matrix_t* A = matrix_LoadCSR("../Tests/dummy/tridiagonal.bin");
		matrix_PrintAsDense( A, "Original coeffient matrix" );

		// Compute matrix bandwidth

		block_t*  x = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
		block_t*  f = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

		block_InitializeToValue( x, __zero ); // solution of the system
		block_InitializeToValue( f, __unit ); // rhs of the system

		#undef _SOLVE_ONLY_WITH_REF_
		#ifdef _SOLVE_ONLY_WITH_REF_
			SolveOriginalSystem( A, x, f);
			matrix_Deallocate( A );
			block_Deallocate( x );
			block_Deallocate( f );
			return 0;
		#endif

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
			sendMatrix(Aij, p+1);

			block_t*  fi  = block_ExtractBlock    ( f, r0, rf );
			block_t*  yi  = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
			block_SetBandwidthValues( fi, A->ku, A->kl );
			block_SetBandwidthValues( yi, A->ku, A->kl );

			sendBlock(fi, p+1);
			sendBlock(yi, p+1);

			/* Add the tips of the yi block to the reduced RHS */
			block_t* yit = recvBlock(p+1);
			block_t* yib = recvBlock(p+1);
			block_AddTipTOReducedRHS( p, S->ku, S->kl, xr, yit );
			block_AddTipTOReducedRHS( p, S->ku, S->kl, xr, yib );

			/* clean up */
			block_Deallocate (fi );
			block_Deallocate (yi );
			block_Deallocate (yit);
			block_Deallocate (yib);
			
			if(p == 0){
				block_t* Vi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
				block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );
				sendBlock(Vi, p+1);
				sendBlock(Bi, p+1);
				printf("Sended Block Vi, Bi to %d\n", p+1);
							
				block_t* Vit = recvBlock(p+1);
				block_t* Vib = recvBlock(p+1);
				printf("Recived Block Vit, Vib from %d\n", p+1);
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );

				block_Deallocate( Bi );
				block_Deallocate( Vi );
				block_Deallocate( Vit);
				block_Deallocate( Vib);
			}
			else if (p == ( S->p -1)){
				block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
				block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );
				sendBlock(Wi, p+1);
				sendBlock(Ci, p+1);

				block_t* Wit = recvBlock(p+1);
				block_t* Wib = recvBlock(p+1);
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );

				block_Deallocate( Ci );
				block_Deallocate( Wi );
				block_Deallocate( Wit);
				block_Deallocate( Wib);
			}
			else{
				block_t* Vi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
				block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );
				sendBlock(Vi, p+1);
				sendBlock(Bi, p+1);
			
				block_t* Vit = recvBlock(p+1);
				block_t* Vib = recvBlock(p+1);			
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );

				block_Deallocate( Bi );
				block_Deallocate( Vi );
				block_Deallocate( Vit);
				block_Deallocate( Vib);
				
				block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
				block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );
				sendBlock(Wi, p+1);
				sendBlock(Ci, p+1);

				block_t* Wit = recvBlock(p+1);
				block_t* Wib = recvBlock(p+1);
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );
				matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );

				block_Deallocate( Ci );
				block_Deallocate( Wi );
				block_Deallocate( Wit);
				block_Deallocate( Wib);
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		/* -------------------------------------------------------------------- */
		/* .. Solution of the reduced system.                                   */
		/* -------------------------------------------------------------------- */

		block_t* yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );  
		fprintf(stderr, "\nSolving reduced linear system\n");
		system_solve ( R->colind, R->rowptr, R->aij, yr->aij, xr->aij, R->n, xr->m);
		block_Print(yr, "Solucion del sistema reducido");


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

			debug("p:%d, obs:%d, obe:%d, rbe:%d, rbs:%d ni:%d",p, obs, obe, rbs, rbe, ni);

			/* allocate pardiso configuration parameters */
			MKL_INT pardiso_conf[64];

			/* extract xi sub-block */
			block_t*  xi  = block_ExtractBlock(x, obs, obe );
			sendBlock(xi, p+1);

			/* extract fi sub-block */
			block_t*  fi  = block_ExtractBlock(f, obs, obe );
			sendBlock(fi, p+1);
			printf("Lets go %d\n", p);
			
			if ( p == 0 ){

				block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
				sendBlock(xt_next, p+1);
				block_Deallocate (xt_next);
			}

			else if ( p == ( S->p -1)){

				block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );
				sendBlock(xb_prev, p+1);
				block_Deallocate (xb_prev);
			}

			else{
				block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);
				sendBlock(xt_next, p+1);
				block_Deallocate (xt_next);
				
				block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );
				sendBlock(xb_prev, p+1);
				block_Deallocate (xb_prev);
			}
			xi = recvBlock(p+1);
			block_AddBlockToRHS(x, xi, obs, obe);
			//directSolver_CleanUp(NULL,NULL,NULL,NULL,NULL, Aij->n, nrhs, &pardiso_conf);
			block_Deallocate    ( xi );
			block_Deallocate 	( fi );
			//matrix_Deallocate	( Aij);

		}
		schedule_Destroy  ( S );
		block_Deallocate  ( yr);
	
		end_t = GetReferenceTime();

		fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);
		block_Print( x, "Solution of the linear system");

		ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
	
		fprintf(stderr, "\nPARDISO REFERENCE SOLUTION...\n");
		SolveOriginalSystem( A, x, f);

		/* -------------------------------------------------------------------- */
		/* .. Clean up. */
		/* -------------------------------------------------------------------- */
		matrix_Deallocate ( A );
		block_Deallocate  ( x );
		block_Deallocate  ( f );
	
		// directSolver_CleanUp( pardiso_conf );



		/* -------------------------------------------------------------------- */
		/* .. Load and initalize the system Ax=f. */
		/* -------------------------------------------------------------------- */
		fprintf(stderr, "\nProgram finished\n");
	}

	else{

		/* -------------------------------------------------------------------- */
		/* .. Factorization Phase. */
		/* -------------------------------------------------------------------- */
		//fprintf(stderr, "Solving %d-th block\n", p);

		
		/* allocate pardiso configuration parameters */
		// void *pardiso_conf = (void*) spike_malloc( ALIGN_INT, 64, sizeof(integer_t));

		sm_schedule_t* S = recvSchedulePacked(master);
		/* compute the limits of the blocks */
		integer_t p = rank -1;
		const integer_t obs = S->n[p];        		/* original system starting row */
		const integer_t obe = S->n[p+1];	  		/* original system ending row   */
		const integer_t rbs = S->r[p];		  		/* reduceed system starting row */
		const integer_t rbe = S->r[p+1];			/* reduced system ending row    */
		const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */
		debug("p:%d, obs:%d, obe:%d, rbe:%d, rbs:%d ni:%d",p, obs, obe, rbs, rbe, ni);

		MKL_INT pardiso_conf[64];

		/* factorize matrix */
		matrix_t* Aij = recvMatrix(master);
		directSolver_Factorize( Aij->colind, Aij->rowptr, Aij->aij, Aij->n, nrhs, &pardiso_conf);

		/* -------------------------------------------------------------------- */
		/* .. Solve Ai * yi = fi                                                */
		/* Extracts the fi portion from f, creates a yi block used as container */
		/* for the solution of the system. Then solves the system.              */
		/* -------------------------------------------------------------------- */
		
		/* solve the system for the RHS value */
		block_t*  fi = recvBlock(master);
		block_t*  yi = recvBlock(master);
		directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, yi->aij, fi->aij, Aij->n, nrhs, &pardiso_conf );

		/* Extract the tips of the yi block */
		block_t* yit = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
		block_t* yib = block_ExtractTip( yi, _BOTTOM_SECTION_, _COLMAJOR_ );
		sendBlock(yit, master);
		sendBlock(yib, master);

		/* clean up */
		block_Deallocate (fi );
		block_Deallocate (yi );
		block_Deallocate (yit);
		block_Deallocate (yib);

		if ( rank == 1 ){
				
			block_t* Vi = recvBlock(master);
			block_t* Bi = recvBlock(master);
			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			sendBlock(Vit, master);
			sendBlock(Vib, master);

			block_t* Bib = block_ExtractTip( Bi, _BOTTOM_SECTION_, _ROWMAJOR_ );

			//block_Deallocate( Vi );
			block_Deallocate( Bi );
			block_Deallocate( Vi);
			block_Deallocate( Vit);
			block_Deallocate( Vib);

			//Here Master Resolve Reduced System
			MPI_Barrier(MPI_COMM_WORLD);
			
			block_t* xi = recvBlock(master);
			block_t* fi = recvBlock(master);
			block_t* xt_next = recvBlock(master);
			
			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
			block_Print(Bib, "Bi Slave");
			cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
				Bib->n,    						/* m - number of rows of A    */
				xt_next->m, 					/* n - number of columns of B */
				Bib->m,    						/* k - number of columns of A */
				-1.0, 							/* alpha                      */
				Bib->aij, 						/* A block                    */
				Bib->n,    						/* lda - first dimension of A */
				xt_next->aij, 					/* B block                    */
				xt_next->n,    					/* ldb - first dimension of B */
				1.0, 							/* beta                       */
				&fi->aij[ni - S->ku[p]], 		/* C block                    */
				ni ); 					 		/* ldc - first dimension of C */
			
			block_Print(xi, "Slave xi after");
			block_Print(fi, "Slave fi after");
			block_Print(xt_next, "Slave xt_next after");
			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );
			block_Print(xi, "Slave xi after");
			sendBlock(xi, master);

			block_Deallocate ( Bib );
			block_Deallocate ( xt_next);
			block_Deallocate ( xi );
			block_Deallocate ( fi );
		}

		else if ( rank == size -1){
			
			block_t* Wi = recvBlock(master);
			block_t* Ci = recvBlock(master);
			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m, &pardiso_conf );


			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			sendBlock(Wit, master);
			sendBlock(Wib, master);
			
			block_t* Cit = block_ExtractTip(Ci, _TOP_SECTION_, _ROWMAJOR_ );			
		
			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);

			//Here Master Resolve Reduced System
			MPI_Barrier(MPI_COMM_WORLD);

			block_t* xi = recvBlock(master);
			block_t* fi = recvBlock(master);
			block_t* xb_prev = recvBlock(master);
			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */ 
			cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
				Cit->n,    						/* m - number of rows of A    */
				xb_prev->m, 					/* n - number of columns of B */
				Cit->m,    						/* k - number of columns of A */
				-1.0, 							/* alpha                      */
				Cit->aij, 						/* A block                    */
				Cit->n,    						/* lda - first dimension of A */
				xb_prev->aij, 					/* B block                    */
				xb_prev->n,    					/* ldb - first dimension of B */
				1.0, 							/* beta                       */
				fi->aij, 			 		    /* C block                    */
				ni );		 					/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );
			sendBlock(xi, master);

			block_Deallocate ( Cit );
			block_Deallocate ( xb_prev);
			block_Deallocate ( xi );
			block_Deallocate ( fi );
		}

		else{
			block_t* Vi = recvBlock(master);
			block_t* Bi = recvBlock(master);
			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			sendBlock(Vit, master);
			sendBlock(Vib, master);

			block_t* Bib = block_ExtractTip( Bi, _BOTTOM_SECTION_, _ROWMAJOR_ );

			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);

			block_t* Wi = recvBlock(master);
			block_t* Ci = recvBlock(master);
			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m, &pardiso_conf );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			sendBlock(Wit, master);
			sendBlock(Wib, master);
			
			block_t* Cit = block_ExtractTip(Ci, _TOP_SECTION_, _ROWMAJOR_ );
			
			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);

			//Here Master Resolve Reduced System
			MPI_Barrier(MPI_COMM_WORLD);
			
			block_t* xi = recvBlock(master);
			block_t* fi = recvBlock(master);
			block_t* xt_next = recvBlock(master);
			
			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
			cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
				Bib->n,    						/* m - number of rows of A    */
				xt_next->m, 					/* n - number of columns of B */
				Bib->m,    						/* k - number of columns of A */
				-1.0, 							/* alpha                      */
				Bib->aij, 						/* A block                    */
				Bib->n,    						/* lda - first dimension of A */
				xt_next->aij, 					/* B block                    */
				xt_next->n,    					/* ldb - first dimension of B */
				1.0, 							/* beta                       */
				&fi->aij[ni - S->ku[p]], 		/* C block                    */
				ni ); 					 		/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );
			//sendBlock(xi, master);

			block_Deallocate ( Bib );
			block_Deallocate ( xt_next); 

			block_t* xb_prev = recvBlock(master);
			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */ 
			cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
				Cit->n,    						/* m - number of rows of A    */
				xb_prev->m, 					/* n - number of columns of B */
				Cit->m,    						/* k - number of columns of A */
				-1.0, 							/* alpha                      */
				Cit->aij, 						/* A block                    */
				Cit->n,    						/* lda - first dimension of A */
				xb_prev->aij, 					/* B block                    */
				xb_prev->n,    					/* ldb - first dimension of B */
				1.0, 							/* beta                       */
				fi->aij, 			 		    /* C block                    */
				ni );		 					/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );
			sendBlock(xi, master);

			block_Deallocate ( Cit );
			block_Deallocate ( xb_prev);
			block_Deallocate ( xi );
			block_Deallocate ( fi );
		
		}

		matrix_Deallocate	( Aij);	
		directSolver_CleanUp(NULL, NULL, NULL, NULL, NULL, Aij->n, nrhs, &pardiso_conf);

	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
