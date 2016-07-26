#include "spike_interfaces.h"

 Error_t sspike_core_host  (const integer_t n,
							const integer_t nnz,
							const integer_t nrhs,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							float     *restrict aij,
							float     *restrict xij,
							float     *restrict bij)
{
	return (SPIKE_SUCCESS);
};


 Error_t dspike_core_host  (const integer_t n,
							const integer_t nnz,
							const integer_t nrhs,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							double     *restrict aij,
							double     *restrict xij,
							double     *restrict bij)
{
	return (SPIKE_SUCCESS);
};


 Error_t cspike_core_host  (const integer_t n,
							const integer_t nnz,
							const integer_t nrhs,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							complex8  *restrict aij,
							complex8  *restrict xij,
							complex8  *restrict bij)
{
	return (SPIKE_SUCCESS);
};



 Error_t zspike_core_host  (const integer_t n,
							const integer_t nnz,
							const integer_t nrhs,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							complex16 *restrict aij,
							complex16 *restrict xij,
							complex16 *restrict bij)
{
	/* non-buffering std error */
	setvbuf(stderr, NULL, _IONBF, 0);

	/* -------------------------------------------------------------------- */
	/* .. Initialize internal structures with external data               . */
	/* -------------------------------------------------------------------- */
	matrix_t *A = matrix_CreateFromComponents (n, nnz, colind, rowptr, (complex_t *restrict) aij );
	block_t  *x = block_CreateFromComponents  (n, nrhs, (complex_t *restrict) xij );
	block_t  *f = block_CreateFromComponents  (n, nrhs, (complex_t *restrict) bij );

	/* -------------------------------------------------------------------- */
	/* .. Local variables.                                                  */
	/* -------------------------------------------------------------------- */
	spike_timer_t start_t;
	spike_timer_t end_t;

	sm_schedule_t *S;

	matrix_t *R;    /* coefficient matrix of the reduced system */
	block_t  *xr;	/* RHS of the reduced system                */
	block_t  *yr;   /* Solution of the reduced system           */


	/* initialize timer */
	start_t = GetReferenceTime();

	/* compute an optimal solving strategy */
	S = spike_solve_analysis( A, nrhs, 20 );

	/* create the reduced sytem in advanced, based on the solving strategy */
	R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
	xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );


	/* -------------------------------------------------------------------- */
	/* .. Factorization Phase. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\n\n----------- Factorization phase ------------------\n\n");

	for(integer_t p=0; p < S->p; p++)
	{
		fprintf(stderr, "\n\tProcessing "_I_"/"_I_" diagonal block\n", p+1, S->p);
		
		const integer_t r0 = S->n[p];
		const integer_t rf = S->n[p+1];

		/* allocate pardiso configuration parameters */
		// void *pardiso_conf = (void*) spike_malloc( ALIGN_INT, 64, sizeof(integer_t));
		MKL_INT pardiso_conf[64];

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
		directSolver_Factorize( Aij->colind, Aij->rowptr, Aij->aij, Aij->n, nrhs, &pardiso_conf);

		/* -------------------------------------------------------------------- */
		/* Solve Ai * yi = fi                                                   */
		/* Extracts the fi portion from f, creates a yi block used as container */
		/* for the solution of the system. Then solves the system.              */
		/* -------------------------------------------------------------------- */
		block_t*  fi  = block_ExtractBlock    ( f, r0, rf );
		block_t*  yi  = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

		block_SetBandwidthValues( fi, A->ku, A->kl );
		block_SetBandwidthValues( yi, A->ku, A->kl );

		/* solve the system for the RHS value */
		directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, yi->aij, fi->aij, Aij->n, nrhs, &pardiso_conf );

		/* Extract the tips of the yi block */
		block_t* yit = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
		block_t* yib = block_ExtractTip( yi, _BOTTOM_SECTION_, _COLMAJOR_ );

		/* Add the tips of the yi block to the reduced RHS */
		block_AddTipTOReducedRHS( p, S->ku, S->kl, xr, yit );
		block_AddTipTOReducedRHS( p, S->ku, S->kl, xr, yib );

		/* clean up */
		block_Deallocate (fi );
		block_Deallocate (yi );
		block_Deallocate (yit);
		block_Deallocate (yib);

		if ( p == 0 ){
			block_t* Vi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );

			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );

			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);
		}
		else if ( p == ( S->p -1)){
			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m, &pardiso_conf );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );

			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );
			
			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);
		}
		else{
			block_t* Vi    = block_CreateEmptyBlock( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi    = matrix_ExtractBlock   (A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Vi->aij, Bi->aij, Aij->n, Vi->m, &pardiso_conf );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );

			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );

			block_Deallocate( Bi );
			block_Deallocate( Vi );
			block_Deallocate( Vit);
			block_Deallocate( Vib);

			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, Wi->aij, Ci->aij, Aij->n, Wi->m, &pardiso_conf );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );

			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );

			block_Deallocate( Ci );
			block_Deallocate( Wi );
			block_Deallocate( Wit);
			block_Deallocate( Wib);
		}

		directSolver_CleanUp(NULL, NULL, NULL, NULL, NULL, Aij->n, nrhs, &pardiso_conf);

		matrix_Deallocate(Aij); 
	}

	/* -------------------------------------------------------------------- */
	/* .. Solution of the reduced system.                                   */
	/* -------------------------------------------------------------------- */

	yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );  
	fprintf(stderr, "\nSolving reduced linear system\n");
	system_solve ( R->colind, R->rowptr, R->aij, yr->aij, xr->aij, R->n, xr->m);


	/* Free some memory, yr and R are not needed anymore */
	block_Deallocate ( xr );
	matrix_Deallocate( R  );

	/* -------------------------------------------------------------------- */
	/* .. Backward substitution phase. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\n\n-------- Backward substitution phase --------------\n\n");

	for(integer_t p=0; p < S->p; p++)
	{
		fprintf(stderr, "\n\tProcessing "_I_"/"_I_" diagonal block\n", p+1, S->p);

		/* compute the limits of the blocks */
		const integer_t obs = S->n[p];        		/* original system starting row */
		const integer_t obe = S->n[p+1];	  		/* original system ending row   */
		const integer_t rbs = S->r[p];		  		/* reduceed system starting row */
		const integer_t rbe = S->r[p+1];			/* reduced system ending row    */
		const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */

		/* allocate pardiso configuration parameters */
		MKL_INT pardiso_conf[64];

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, obs, obe, obs, obe);
		directSolver_Factorize( Aij->colind, Aij->rowptr, Aij->aij, Aij->n, nrhs, &pardiso_conf);

		/* extract xi sub-block */
		block_t*  xi  = block_ExtractBlock(x, obs, obe );

		/* extract fi sub-block */
		block_t*  fi  = block_ExtractBlock(f, obs, obe );
			
		if ( p == 0 ){

			block_t* Bi  = matrix_ExtractBlock ( A, obe - S->ku[p], obe, obe, obe + S->ku[p], _WHOLE_SECTION_ );
			block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);

			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi using cblas_?gemm*/
			gemm( _COLMAJOR_, _NOTRANSPOSE_, _NOTRANSPOSE_,
				Bi->n,    						/* m - number of rows of A    */
				xt_next->m, 					/* n - number of columns of B */
				Bi->m,    						/* k - number of columns of A */
				__nunit, 						/* alpha                      */
				Bi->aij, 						/* A block                    */
				Bi->n,    						/* lda - first dimension of A */
				xt_next->aij, 					/* B block                    */
				xt_next->n,    					/* ldb - first dimension of B */
				__punit, 						/* beta                       */
				&fi->aij[ni - S->ku[p]], 		/* C block                    */
				ni ); 					 		/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );

			block_Deallocate ( Bi );
			block_Deallocate ( xt_next); 
		}
		else if ( p == ( S->p -1)){

			block_t* Ci  = matrix_ExtractBlock ( A, obs, obs + S->kl[p], obs - S->kl[p], obs, _WHOLE_SECTION_ );
			block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );

			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi using cblas_?gemm*/
			gemm( _COLMAJOR_, _NOTRANSPOSE_, _NOTRANSPOSE_,
				Ci->n,    						/* m - number of rows of A    */
				xb_prev->m, 					/* n - number of columns of B */
				Ci->m,    						/* k - number of columns of A */
				__nunit,						/* alpha                      */
				Ci->aij, 						/* A block                    */
				Ci->n,    						/* lda - first dimension of A */
				xb_prev->aij, 					/* B block                    */
				xb_prev->n,    					/* ldb - first dimension of B */
				__punit,						/* beta                       */
				fi->aij, 			 		    /* C block                    */
				ni );		 					/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );

			block_Deallocate ( Ci );
			block_Deallocate ( xb_prev); 
		
		}
		else{

			block_t* Bi  = matrix_ExtractBlock ( A, obe - S->ku[p], obe, obe, obe + S->ku[p], _WHOLE_SECTION_ );
			block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p+1]);

			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi using cblas_?gemm*/
			gemm( _COLMAJOR_, _NOTRANSPOSE_, _NOTRANSPOSE_,
				Bi->n,    						/* m - number of rows of A    */
				xt_next->m, 					/* n - number of columns of B */
				Bi->m,    						/* k - number of columns of A */
				__nunit,						/* alpha                      */
				Bi->aij, 						/* A block                    */
				Bi->n,    						/* lda - first dimension of A */
				xt_next->aij, 					/* B block                    */
				xt_next->n,    					/* ldb - first dimension of B */
				__punit,						/* beta                       */
				&fi->aij[ni - S->ku[p]], 		/* C block                    */
				ni ); 					 		/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );

			block_Deallocate ( Bi );
			block_Deallocate ( xt_next); 

			block_t* Ci  = matrix_ExtractBlock ( A, obs, obs + S->kl[p], obs - S->kl[p], obs, _WHOLE_SECTION_ );			
			block_t* xb_prev = block_ExtractBlock ( yr, rbs - S->kl[p], rbs );

			/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi using cblas_?gemm*/
			gemm( _COLMAJOR_, _NOTRANSPOSE_, _NOTRANSPOSE_,
				Ci->n,    						/* m - number of rows of A    */
				xb_prev->m, 					/* n - number of columns of B */
				Ci->m,    						/* k - number of columns of A */
				__nunit,						/* alpha                      */
				Ci->aij, 						/* A block                    */
				Ci->n,    						/* lda - first dimension of A */
				xb_prev->aij, 					/* B block                    */
				xb_prev->n,    					/* ldb - first dimension of B */
				__punit,						/* beta                       */
				fi->aij, 			 		    /* C block                    */
				ni );		 					/* ldc - first dimension of C */

			directSolver_ApplyFactorToRHS( Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi->aij, Aij->n, xi->m, &pardiso_conf );

			block_Deallocate ( Ci );
			block_Deallocate ( xb_prev);
		}

		block_AddBlockToRHS(x, xi, obs, obe);

		directSolver_CleanUp(NULL,NULL,NULL,NULL,NULL, Aij->n, nrhs, &pardiso_conf);
		block_Deallocate    ( xi );
		block_Deallocate 	( fi );
		matrix_Deallocate	( Aij);

	}
	fprintf(stderr, "\n\n----------- Release memory phase ------------------\n\n");

	schedule_Destroy  ( S );
	block_Deallocate  ( yr);
	
	end_t = GetReferenceTime();

	fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);

	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
	
	return (SPIKE_SUCCESS);
};