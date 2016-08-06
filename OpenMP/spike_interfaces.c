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


 Error_t zspike_core_host  (const integer_t n,
							const integer_t nnz,
							const integer_t nrhs,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							complex16 *restrict aij,
							complex16 *restrict xij,
							complex16 *restrict bij)
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



 Error_t dspike_core_host  (const integer_t n,
							const integer_t nnz,
							const integer_t nrhs,
							integer_t *restrict colind,
							integer_t *restrict rowptr,
							double *restrict aij,
							double *restrict xij,
							double *restrict bij,
							const int partitions)
{
	// TODO: write a check params function 
	if ( nnz < 0 ) {
		fprintf(stderr, "\n%s: nnz must be a positive number (consider buffer overflow)\n", __FUNCTION__ );
		abort();
	}

	// /gpfs/scratch/bsc21/bsc21253/tests/POC_FMD_3D_MZANZI_003/exec_input>
	// /gpfs/scratch/bsc21/bsc21225/BSITLocal/trunk/system/main/kernel/bin/em.iso.gp.fm kernel.fm.prm.freq0.05Hz.000001
	
	fprintf(stderr, "\n  SPIKE direct-direct solver\n");
	/* non-buffering std error */
	// setvbuf(stderr, NULL, _IONBF, 0);

	/* -------------------------------------------------------------------- */
	/* .. Initialize internal structures with external data               . */
	/* -------------------------------------------------------------------- */
	matrix_t *A = matrix_CreateFromComponents (n, nnz, colind, rowptr, (complex_t *restrict) aij );
	block_t  *x = block_CreateFromComponents  (n, nrhs, (complex_t *restrict) xij );
	block_t  *f = block_CreateFromComponents  (n, nrhs, (complex_t *restrict) bij );

	fprintf(stderr, "\nCreated internal data structures\n");

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
	S = spike_solve_analysis( A, nrhs, partitions );

	/* create the reduced sytem in advanced, based on the solving strategy */
	R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
	fprintf(stderr, "Created reduced system.\n");

	xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );
	fprintf(stderr, "Created reduced system rhs.\n");

	/* -------------------------------------------------------------------- */
	/* .. Factorization Phase. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\n\n----------- Factorization phase ------------------\n\n");

	for(integer_t p=0; p < S->p; p++)
	{
		fprintf(stderr, "\n\tProcessing "_I_"/"_I_" diagonal block\n", p+1, S->p);
		
		const integer_t r0 = S->n[p];
		const integer_t rf = S->n[p+1];

		/* Set up solver handler */
		DirectSolverHander_t *handler = directSolver_CreateHandler();
		
		directSolver_Configure(handler);

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
		fprintf(stderr, "Sub-matrix Aij extracted\n");
		
		directSolver_Factorize( handler, 
								Aij->n,
								Aij->nnz, 
								Aij->colind, 
								Aij->rowptr, 
								Aij->aij);

		fprintf(stderr, "Matrix factorized\n");

		/* -------------------------------------------------------------------- */
		/* Solve Ai * yi = fi                                                   */
		/* Extracts the fi portion from f, creates a yi block used as container */
		/* for the solution of the system. Then solves the system.              */
		/* -------------------------------------------------------------------- */
		block_t*  fi  = block_ExtractBlock    ( f, r0, rf );
		block_t*  yi  = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
		fprintf(stderr, "yi block created\n");

		block_SetBandwidthValues( fi, A->ku, A->kl );
		block_SetBandwidthValues( yi, A->ku, A->kl );

		/* solve the system for the RHS value */
		directSolver_SolveForRHS( handler, nrhs, yi->aij, fi->aij );
		fprintf(stderr, "Solved for RHS\n");


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

			/* solve Aij * Vi = Bi */
			directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij );
			block_Deallocate( Bi );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );
			block_Deallocate( Vit);

			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );
			block_Deallocate( Vi );

			block_Deallocate( Vib);
		}
		else if ( p == ( S->p -1)){
			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			/* solve Aij * Wi = Ci */
			directSolver_SolveForRHS( handler, Wi->m, Wi->aij, Ci->aij );
			block_Deallocate( Ci );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );
			block_Deallocate( Wit);

			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );
			block_Deallocate( Wib);
			
			block_Deallocate( Wi );
		}
		else{
			block_t* Vi    = block_CreateEmptyBlock( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
			block_t* Bi    = matrix_ExtractBlock   (A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

			/* solve Aij * Vi = Bi */
			directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij );
			block_Deallocate( Bi );

			block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vit );
			block_Deallocate( Vit);

			block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Vib );
			block_Deallocate( Vib);

			block_Deallocate( Vi );

			block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
			block_t* Ci = matrix_ExtractBlock(A, r0, rf, r0 - A->kl, r0, _W_BLOCK_ );

			/* solve Aij * Wi = Ci */
			directSolver_SolveForRHS( handler, Wi->m, Wi->aij, Ci->aij );
			block_Deallocate( Ci );

			block_t* Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wit );
			block_Deallocate( Wit);

			block_t* Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
			matrix_AddTipToReducedMatrix( S->p, p, S->n, S->ku, S->kl, R, Wib );
			block_Deallocate( Wib);

			block_Deallocate( Wi );
		}

		directSolver_ShowStatistics(handler);
		directSolver_Finalize(handler);

		matrix_Deallocate(Aij); 
	}

	/* -------------------------------------------------------------------- */
	/* .. Solution of the reduced system.                                   */
	/* -------------------------------------------------------------------- */

	yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );  
	fprintf(stderr, "\nSolving reduced linear system\n");
	directSolver_Host_Solve ( R->n, R->nnz, xr->m, R->colind, R->rowptr, R->aij, yr->aij, xr->aij );

	/* compute residual */
	ComputeResidualOfLinearSystem( R->colind, R->rowptr, R->aij, yr->aij, xr->aij, R->n, yr->m );

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
		DirectSolverHander_t *handler = directSolver_CreateHandler();

		directSolver_Configure( handler );

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, obs, obe, obs, obe);
		directSolver_Factorize( handler, Aij->n, Aij->nnz, Aij->colind, Aij->rowptr, Aij->aij);

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

			/* Solve Aij * ( f - Bi * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

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

			block_Deallocate ( Ci );
			block_Deallocate ( xb_prev); 

			/* Solve Aij * ( f - Ci * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

		
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

			block_Deallocate ( Bi );
			block_Deallocate ( xt_next); 

			/* Solve Aij * ( f - Bi * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

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

			block_Deallocate ( Ci );
			block_Deallocate ( xb_prev);

			/* Solve Aij * ( f - Bi * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

		}

		block_AddBlockToRHS(x, xi, obs, obe);

		directSolver_ShowStatistics(handler);
		directSolver_Finalize(handler);

		block_Deallocate    ( xi );
		block_Deallocate 	( fi );
		matrix_Deallocate	( Aij);

	}

	fprintf(stderr, "\n\n----------- Release memory phase ------------------\n\n");

	block_Deallocate  ( yr );
	schedule_Destroy  ( S  );
	

	fprintf(stderr, "Internal SPIKE residual\n");
	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
	
	/* deallocate temporal structures */
	spike_nullify( A );
	spike_nullify( x );
	spike_nullify( f );


	end_t = GetReferenceTime();

	fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);

	
	return (SPIKE_SUCCESS);
};

 Error_t dspike_core_host_blocking (
 	const integer_t n,
	const integer_t nnz,
	const integer_t nrhs,
	integer_t *restrict colind,
	integer_t *restrict rowptr,
	double *restrict aij,
	double *restrict xij,
	double *restrict bij,
	const int partitions )
{
	// TODO: write a check params function 
	if ( nnz < 0 ) {
		fprintf(stderr, "\n%s: nnz must be a positive number (consider buffer overflow)\n", __FUNCTION__ );
		abort();
	}
	
	/* define column blocking size */
	const integer_t COLBLOCKINGDIST = 10;

	fprintf(stderr, "\n  SPIKE direct-direct solver (low-level, blocking implementation)\n");

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
	integer_t     col = 0;

	sm_schedule_t *S;

	matrix_t *R;    /* coefficient matrix of the reduced system */
	block_t  *xr;	/* RHS of the reduced system                */
	block_t  *yr;   /* Solution of the reduced system           */


	/* initialize timer */
	start_t = GetReferenceTime();

	/* compute an optimal solving strategy */
	S = spike_solve_analysis( A, nrhs, partitions );

	/* create the reduced sytem in advanced, based on the solving strategy */
	R  = matrix_CreateEmptyReducedSystem ( S->p, S->n, S->ku, S->kl);
	fprintf(stderr, "Created reduced system.\n");

	xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );
	fprintf(stderr, "Created reduced system rhs.\n");

	/* -------------------------------------------------------------------- */
	/* .. Factorization Phase. */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\n\n----------- Factorization phase ------------------\n\n");

	for(integer_t p=0; p < S->p; p++)
	{
		fprintf(stderr, "\n\tProcessing "_I_"/"_I_" diagonal block\n", p+1, S->p);
		
		const integer_t r0 = S->n[p];
		const integer_t rf = S->n[p+1];

		/* Set up solver handler */
		DirectSolverHander_t *handler = directSolver_CreateHandler();
		
		directSolver_Configure(handler);

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf);
		fprintf(stderr, "Sub-matrix Aij extracted\n");
		
		directSolver_Factorize( handler, 
								Aij->n,
								Aij->nnz, 
								Aij->colind, 
								Aij->rowptr, 
								Aij->aij);

		fprintf(stderr, "Matrix factorized\n");

		/* -------------------------------------------------------------------- */
		/* Solve Ai * yi = fi                                                   */
		/* Extracts the fi portion from f, creates a yi block used as container */
		/* for the solution of the system. Then solves the system.              */
		/* -------------------------------------------------------------------- */
		if ( nrhs < COLBLOCKINGDIST ) {
			/* blocking buffer */
			block_t *fi = block_CreateEmptyBlock( rf - r0, nrhs, A->ku, A->kl, _RHS_BLOCK_, _WHOLE_SECTION_);
			block_t *yi = block_CreateEmptyBlock( rf - r0, nrhs, A->ku, A->kl, _RHS_BLOCK_, _WHOLE_SECTION_);


			block_InitializeToValue( yi, __zero  ); // TODO: optimize using memset

			/* Extract the fi sub-block */
			block_ExtractBlock_blocking ( fi, f, r0, rf, 0, nrhs );

			/* solve the system for the RHS value */
			directSolver_SolveForRHS ( handler, nrhs, yi->aij, fi->aij );

			/* extract the yit tip using fi as buffer, then, add it to the reduced system RHS */
			block_ExtractTip_blocking          ( fi, yi, 0, nrhs, _TOP_SECTION_, _COLMAJOR_ );
			block_AddTipTOReducedRHS_blocking  ( p, 0, nrhs, S->ku, S->kl, xr, fi );
			
			/* extract the yib tip using fi as buffer, then, add it to the reduced system RHS */
			block_ExtractTip_blocking          ( fi, yi, 0, nrhs, _BOTTOM_SECTION_, _COLMAJOR_ );
			block_AddTipTOReducedRHS_blocking  ( p, 0, nrhs, S->ku, S->kl, xr, fi );

			/* clean up */
			block_Deallocate (fi );
			block_Deallocate (yi );
		}
		else{
			/* blocking buffer */
			block_t *fi = block_CreateEmptyBlock( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _RHS_BLOCK_, _WHOLE_SECTION_);
			block_t *yi = block_CreateEmptyBlock( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _RHS_BLOCK_, _WHOLE_SECTION_);


			for(col = 0; (col + COLBLOCKINGDIST) < nrhs; col += COLBLOCKINGDIST ) {

				block_InitializeToValue( yi, __zero  ); // TODO: optimize using memset

				/* Extract the fi sub-block */
				block_ExtractBlock_blocking ( fi, f, r0, rf, col, col + COLBLOCKINGDIST );

				/* solve the system for the RHS value */
				directSolver_SolveForRHS ( handler, COLBLOCKINGDIST, yi->aij, fi->aij );

				/* extract the yit tip using fi as buffer, then, add it to the reduced system RHS */
				block_ExtractTip_blocking          ( fi, yi, 0, COLBLOCKINGDIST, _TOP_SECTION_, _COLMAJOR_ );
				block_AddTipTOReducedRHS_blocking  ( p, col, col + COLBLOCKINGDIST, S->ku, S->kl, xr, fi );
				
				/* extract the yib tip using fi as buffer, then, add it to the reduced system RHS */
				block_ExtractTip_blocking          ( fi, yi, 0, COLBLOCKINGDIST, _BOTTOM_SECTION_, _COLMAJOR_ );
				block_AddTipTOReducedRHS_blocking  ( p, col, col + COLBLOCKINGDIST, S->ku, S->kl, xr, fi );
			}
		
			if ( col < nrhs ) {
				block_InitializeToValue( yi, __zero  ); // TODO: optimize using memset

				/* Extract the fi sub-block */
				block_ExtractBlock_blocking ( fi, f, r0, rf, col, nrhs );

				/* solve the system for the RHS value */
				directSolver_SolveForRHS ( handler, nrhs - col , yi->aij, fi->aij );

				/* extract the yit tip using fi as buffer, then, add it to the reduced system RHS */
				block_ExtractTip_blocking          ( fi, yi, 0, nrhs - col, _TOP_SECTION_, _COLMAJOR_ );
				block_AddTipTOReducedRHS_blocking  ( p, col, nrhs, S->ku, S->kl, xr, fi );
				
				/* extract the yib tip using fi as buffer, then, add it to the reduced system RHS */
				block_ExtractTip_blocking          ( fi, yi, 0, nrhs - col, _BOTTOM_SECTION_, _COLMAJOR_ );
				block_AddTipTOReducedRHS_blocking  ( p, col, nrhs, S->ku, S->kl, xr, fi );
			}

			/* clean up */
			block_Deallocate (fi );
			block_Deallocate (yi );
		}

		/* First partition, factorize only Bi -> Vi */
		if ( p == 0 ){

			if ( A->ku < COLBLOCKINGDIST ) {
				/* blocking buffer */
				block_t* Vi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
				block_t* Bi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );

				block_InitializeToValue( Bi, __zero  ); // TODO: optimize using memset
			
				/* Extract the Bi sub-block */
				matrix_ExtractBlock_blocking ( A, Bi, r0, rf, rf, rf + A->ku, _V_BLOCK_ );


				/* solve Aij * Vi = Bi */
				directSolver_SolveForRHS( handler, A->ku, Vi->aij, Bi->aij );

				/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking   ( Bi, Vi, 0, A->ku, _TOP_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->ku, S->n, S->ku, S->kl, R, Bi );

				/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking    ( Bi, Vi, 0, A->ku, _BOTTOM_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->ku, S->n, S->ku, S->kl, R, Bi );

				/* clean up */
				block_Deallocate( Vi );
				block_Deallocate( Bi );
			}
			else{
				/* blocking buffer */
				block_t* Vi = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
				block_t* Bi = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );

				for(col = 0; (col + COLBLOCKINGDIST) < A->ku; col += COLBLOCKINGDIST ) {
					block_InitializeToValue( Bi, __zero  ); // TODO: optimize using memset
			
					/* Extract the Bi sub-block */
					matrix_ExtractBlock_blocking ( A, Bi, r0, rf, rf + col, rf + col + COLBLOCKINGDIST, _V_BLOCK_ );

					/* solve Aij * Vi = Bi */
					directSolver_SolveForRHS( handler, COLBLOCKINGDIST, Vi->aij, Bi->aij );

					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking   ( Bi, Vi, 0, COLBLOCKINGDIST, _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Bi );


					/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking    ( Bi, Vi, 0, COLBLOCKINGDIST, _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Bi );
				}

				if ( col < A->ku ) {
					/* blocking buffer */
					block_InitializeToValue( Bi, __zero  ); // TODO: optimize using memset
					block_InitializeToValue( Vi, __zero  ); // TODO: optimize using memset

					/* Extract the Bi sub-block */
					matrix_ExtractBlock_blocking ( A, Bi, r0, rf, rf + col, rf + A->ku, _V_BLOCK_ );

					/* solve Aij * Vi = Bi */
					directSolver_SolveForRHS( handler, A->ku - col, Vi->aij, Bi->aij );
					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking   ( Bi, Vi, 0, A->ku - col, _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->ku, S->n, S->ku, S->kl, R, Bi );

					/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking    ( Bi, Vi, 0, A->ku - col, _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->ku, S->n, S->ku, S->kl, R, Bi );
				}

				/* clean up */
				block_Deallocate( Vi );
				block_Deallocate( Bi );

			}
		}

		else if ( p == ( S->p -1)){

			if ( A->kl < COLBLOCKINGDIST ) {

				block_t* Ci = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
				block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );

				block_InitializeToValue( Ci, __zero  ); // TODO: optimize using memset

				/* Extract the Ci sub-block */
				matrix_ExtractBlock_blocking (A, Ci, r0, rf,  r0 - A->kl, r0, _W_BLOCK_ );    // TODO!!!

				/* solve Aij * Wi = Ci */
				directSolver_SolveForRHS( handler, A->kl, Wi->aij, Ci->aij );

				/* extract the Wit tip using Ci as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking ( Ci, Wi, 0, A->kl, _TOP_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->kl, S->n, S->ku, S->kl, R, Ci );

				/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking ( Ci, Wi, 0, A->kl, _BOTTOM_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->kl, S->n, S->ku, S->kl, R, Ci );
			
				/* clean up */			
				block_Deallocate( Wi );
				block_Deallocate( Ci );
			}
			else {
				/* blocking buffer */
				block_t* Wi = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
				block_t* Ci = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );

				for(col = 0; (col + COLBLOCKINGDIST) < A->kl; col += COLBLOCKINGDIST ) {
					block_InitializeToValue( Ci, __zero  ); // TODO: optimize using memset

					/* Extract the Ci sub-block */
					matrix_ExtractBlock_blocking (A, Ci, r0, rf, (r0 - A->kl) + col, (r0 - A->kl) + col + COLBLOCKINGDIST, _W_BLOCK_ );    // TODO!!!

					/* solve Aij * Wi = Ci */
					directSolver_SolveForRHS( handler, COLBLOCKINGDIST, Wi->aij, Ci->aij );

					/* extract the Wit tip using Ci as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, COLBLOCKINGDIST, _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Ci );

					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, COLBLOCKINGDIST, _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Ci );
				}

				if ( col < A->kl ) {
					block_InitializeToValue( Ci, __zero  ); // TODO: optimize using memset

					/* Extract the Ci sub-block */
					matrix_ExtractBlock_blocking (A, Ci, r0, rf, (r0 - A->kl) + col, r0, _W_BLOCK_ );    // TODO!!!

					/* solve Aij * Wi = Ci */
					directSolver_SolveForRHS( handler, A->kl - col, Wi->aij, Ci->aij );

					/* extract the Wit tip using Ci as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, (A->kl - col), _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->kl, S->n, S->ku, S->kl, R, Ci );

					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, (A->kl - col), _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->kl, S->n, S->ku, S->kl, R, Ci );
				}

				block_Deallocate( Ci );
				block_Deallocate( Wi );
			}
		}


		else{

			if ( A->ku < COLBLOCKINGDIST ) {
				/* blocking buffer */
				block_t* Vi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
				block_t* Bi = block_CreateEmptyBlock ( rf - r0, A->ku, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );

				block_InitializeToValue( Bi, __zero  ); // TODO: optimize using memset
			
				/* Extract the Bi sub-block */
				matrix_ExtractBlock_blocking ( A, Bi, r0, rf, rf, rf + A->ku, _V_BLOCK_ );


				/* solve Aij * Vi = Bi */
				directSolver_SolveForRHS( handler, A->ku, Vi->aij, Bi->aij );

				/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking   ( Bi, Vi, 0, A->ku, _TOP_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->ku, S->n, S->ku, S->kl, R, Bi );

				/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking    ( Bi, Vi, 0, A->ku, _BOTTOM_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->ku, S->n, S->ku, S->kl, R, Bi );

				/* clean up */
				block_Deallocate( Vi );
				block_Deallocate( Bi );
			}
			else{
				/* blocking buffer */
				block_t* Vi = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );
				block_t* Bi = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _V_BLOCK_, _WHOLE_SECTION_ );

				for(col = 0; (col + COLBLOCKINGDIST) < A->ku; col += COLBLOCKINGDIST ) {
					block_InitializeToValue( Bi, __zero  ); // TODO: optimize using memset
			
					/* Extract the Bi sub-block */
					matrix_ExtractBlock_blocking ( A, Bi, r0, rf, rf + col, rf + col + COLBLOCKINGDIST, _V_BLOCK_ );

					/* solve Aij * Vi = Bi */
					directSolver_SolveForRHS( handler, COLBLOCKINGDIST, Vi->aij, Bi->aij );

					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking   ( Bi, Vi, 0, COLBLOCKINGDIST, _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Bi );


					/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking    ( Bi, Vi, 0, COLBLOCKINGDIST, _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Bi );
				}

				if ( col < A->ku ) {
					/* blocking buffer */
					block_InitializeToValue( Bi, __zero  ); // TODO: optimize using memset
					block_InitializeToValue( Vi, __zero  ); // TODO: optimize using memset

					/* Extract the Bi sub-block */
					matrix_ExtractBlock_blocking ( A, Bi, r0, rf, rf + col, rf + A->ku, _V_BLOCK_ );

					/* solve Aij * Vi = Bi */
					directSolver_SolveForRHS( handler, A->ku - col, Vi->aij, Bi->aij );
					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking   ( Bi, Vi, 0, A->ku - col, _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->ku, S->n, S->ku, S->kl, R, Bi );

					/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking    ( Bi, Vi, 0, A->ku - col, _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->ku, S->n, S->ku, S->kl, R, Bi );
				}

				/* clean up */
				block_Deallocate( Vi );
				block_Deallocate( Bi );

			}

			if ( A->kl < COLBLOCKINGDIST ) {

				block_t* Ci = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
				block_t* Wi = block_CreateEmptyBlock( rf - r0, A->kl, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );

				block_InitializeToValue( Ci, __zero  ); // TODO: optimize using memset

				/* Extract the Ci sub-block */
				matrix_ExtractBlock_blocking (A, Ci, r0, rf,  r0 - A->kl, r0, _W_BLOCK_ );    // TODO!!!

				/* solve Aij * Wi = Ci */
				directSolver_SolveForRHS( handler, A->kl, Wi->aij, Ci->aij );

				/* extract the Wit tip using Ci as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking ( Ci, Wi, 0, A->kl, _TOP_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->kl, S->n, S->ku, S->kl, R, Ci );

				/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
				block_ExtractTip_blocking ( Ci, Wi, 0, A->kl, _BOTTOM_SECTION_, _ROWMAJOR_ );
				matrix_AddTipToReducedMatrix_blocking( S->p, p, 0, A->kl, S->n, S->ku, S->kl, R, Ci );
			
				/* clean up */			
				block_Deallocate( Wi );
				block_Deallocate( Ci );
			}
			else {
				/* blocking buffer */
				block_t* Wi = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );
				block_t* Ci = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, A->ku, A->kl, _W_BLOCK_, _WHOLE_SECTION_ );

				for(col = 0; (col + COLBLOCKINGDIST) < A->kl; col += COLBLOCKINGDIST ) {
					block_InitializeToValue( Ci, __zero  ); // TODO: optimize using memset

					/* Extract the Ci sub-block */
					matrix_ExtractBlock_blocking (A, Ci, r0, rf, (r0 - A->kl) + col, (r0 - A->kl) + col + COLBLOCKINGDIST, _W_BLOCK_ );    // TODO!!!

					/* solve Aij * Wi = Ci */
					directSolver_SolveForRHS( handler, COLBLOCKINGDIST, Wi->aij, Ci->aij );


					/* extract the Wit tip using Ci as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, COLBLOCKINGDIST, _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Ci );

					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, COLBLOCKINGDIST, _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, col + COLBLOCKINGDIST, S->n, S->ku, S->kl, R, Ci );
				}

				if ( col < A->kl ) {
					block_InitializeToValue( Ci, __zero  ); // TODO: optimize using memset

					/* Extract the Ci sub-block */
					matrix_ExtractBlock_blocking (A, Ci, r0, rf, (r0 - A->kl) + col, r0, _W_BLOCK_ );    // TODO!!!

					/* solve Aij * Wi = Ci */
					directSolver_SolveForRHS( handler, A->kl - col, Wi->aij, Ci->aij );

					/* extract the Wit tip using Ci as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, (A->kl - col), _TOP_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->kl, S->n, S->ku, S->kl, R, Ci );

					/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
					block_ExtractTip_blocking ( Ci, Wi, 0, (A->kl - col), _BOTTOM_SECTION_, _ROWMAJOR_ );
					matrix_AddTipToReducedMatrix_blocking( S->p, p, col, A->kl, S->n, S->ku, S->kl, R, Ci );
				}

				block_Deallocate( Ci );
				block_Deallocate( Wi );
			}


		}

		directSolver_ShowStatistics(handler);
		directSolver_Finalize(handler);

		matrix_Deallocate(Aij); 
	}

	/* -------------------------------------------------------------------- */
	/* .. Solution of the reduced system.                                   */
	/* -------------------------------------------------------------------- */
	yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );  
	fprintf(stderr, "\nSolving reduced linear system\n");
	directSolver_Host_Solve ( R->n, R->nnz, xr->m, R->colind, R->rowptr, R->aij, yr->aij, xr->aij );

	/* compute residual */
	// ComputeResidualOfLinearSystem( R->colind, R->rowptr, R->aij, yr->aij, xr->aij, R->n, yr->m );

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
		DirectSolverHander_t *handler = directSolver_CreateHandler();

		directSolver_Configure( handler );

		/* factorize matrix */
		matrix_t* Aij = matrix_ExtractMatrix(A, obs, obe, obs, obe);
		directSolver_Factorize( handler, Aij->n, Aij->nnz, Aij->colind, Aij->rowptr, Aij->aij);

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

			/* Solve Aij * ( f - Bi * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

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

			block_Deallocate ( Ci );
			block_Deallocate ( xb_prev); 

			/* Solve Aij * ( f - Ci * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

		
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

			block_Deallocate ( Bi );
			block_Deallocate ( xt_next); 

			/* Solve Aij * ( f - Bi * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

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

			block_Deallocate ( Ci );
			block_Deallocate ( xb_prev);

			/* Solve Aij * ( f - Bi * xt ) */
			directSolver_SolveForRHS( handler, xi->m, xi->aij, fi->aij );

		}

		block_AddBlockToRHS(x, xi, obs, obe);

		directSolver_ShowStatistics(handler);
		directSolver_Finalize(handler);

		block_Deallocate    ( xi );
		block_Deallocate 	( fi );
		matrix_Deallocate	( Aij);

	}

	fprintf(stderr, "\n\n----------- Release memory phase ------------------\n\n");

	block_Deallocate  ( yr );
	schedule_Destroy  ( S  );
	
	block_Print( x, "SPIKE solution");

	fprintf(stderr, "Internal SPIKE residual\n");
	ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x->aij, f->aij, A->n, nrhs);
	
	/* deallocate temporal structures */
	spike_nullify( A );
	spike_nullify( x );
	spike_nullify( f );


	end_t = GetReferenceTime();

	fprintf(stderr, "\nSPIKE solver took %.6lf seconds", end_t - start_t);

	
	return (SPIKE_SUCCESS);
};
