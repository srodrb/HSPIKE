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

#include "spike_interfaces.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DirectSolverHander_t *solver_Init(void)
{
	DirectSolverHander_t *handler = (DirectSolverHander_t*) spike_malloc( ALIGN_INT, 1, sizeof(DirectSolverHander_t));

	return (handler);
};


Error_t spike_pdgssvx_factor(integer_t nprocs,
	superlumt_options_t *superlumt_options,
	SuperMatrix *A, 
	integer_t *perm_c,
	integer_t *perm_r,
	equed_t *equed,
	double *R,
	double *C,
	SuperMatrix *L,
	SuperMatrix *U,
	double *recip_pivot_growth, 
	double *rcond, 
	superlu_memusage_t *superlu_memusage,
	integer_t *info)
{
    NCformat  	*Astore;
    SuperMatrix *AA; /* A in NC format used by the factorization routine.*/
    SuperMatrix  AC; /* Matrix postmultiplied by Pc */

    integer_t       n = A->nrow;
    integer_t       ldb = n;
    integer_t       ldx = n;
    
    
    integer_t     	colequ, equil, dofact, notran, rowequ;
    char      		norm[1];
    trans_t   		trant;
    integer_t     	j, info1;
    int 			i;
    double 			amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
    integer_t       relax, panel_size;
    Gstat_t   		Gstat;
    double    		t0;      /* temporary time */
    double    		*utime;
    flops_t   		*ops, flopcnt;
   
    /* External functions */
    extern real_t dlangs     (char *, SuperMatrix *);
    extern real_t dlamch_    (char *);
    extern void   sp_colorder(SuperMatrix *,
    						  integer_t *,
    						  superlumt_options_t *,
    						  SuperMatrix *);

    Astore 					  = A->Store;
    superlumt_options->perm_c = perm_c;
    superlumt_options->perm_r = perm_r;

    *info = 0;
    dofact = (superlumt_options->fact == DOFACT);
    equil = (superlumt_options->fact == EQUILIBRATE);
    notran = (superlumt_options->trans == NOTRANS);

    if (dofact || equil) {
    	*equed = NOEQUIL;
    	rowequ = FALSE;
    	colequ = FALSE;
    } else {
    	rowequ = (*equed == ROW) || (*equed == BOTH);
    	colequ = (*equed == COL) || (*equed == BOTH);
    	smlnum = dlamch_("Safe minimum");
    	bignum = 1. / smlnum;
    }

    /* ------------------------------------------------------------
       Test the input parameters.
       ------------------------------------------------------------*/
    if ( nprocs <= 0 ) *info = -1;

    else if ( (!dofact && !equil && (superlumt_options->fact != FACTORED))
	      || (!notran && (superlumt_options->trans != TRANS) && 
		 (superlumt_options->trans != CONJ))
	      || (superlumt_options->refact != YES && 
		  superlumt_options->refact != NO)
	      || (superlumt_options->usepr != YES &&
		  superlumt_options->usepr != NO)
	      || superlumt_options->lwork < -1 )

        *info = -2;
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
	      (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
	      A->Dtype != SLU_D || A->Mtype != SLU_GE )

		*info = -3;
    
    else if ((superlumt_options->fact == FACTORED) && 
	     !(rowequ || colequ || (*equed == NOEQUIL)))

    	*info = -6;
    

    else {
		if (rowequ) {
		    rcmin = bignum;
		    rcmax = 0.;
		    for (j = 0; j < A->nrow; ++j) {
				rcmin = SUPERLU_MIN(rcmin, R[j]);
				rcmax = SUPERLU_MAX(rcmax, R[j]);
		    }
		    if (rcmin <= 0.) 
		    	*info = -7;
		    else if ( A->nrow > 0)
				rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
		    else
		    	rowcnd = 1.;
		}

		if (colequ && *info == 0) {
		    rcmin = bignum;
		    rcmax = 0.;
		    for (j = 0; j < A->nrow; ++j) {
				rcmin = SUPERLU_MIN(rcmin, C[j]);
				rcmax = SUPERLU_MAX(rcmax, C[j]);
		    }
		    if (rcmin <= 0.)
		    	*info = -8;
		    else if (A->nrow > 0)
				colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
		    else
		    	colcnd = 1.;
		}

		//if (*info == 0) {
		//    if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
		//	      B->Stype != SLU_DN || B->Dtype != SLU_D || 
		//	      B->Mtype != SLU_GE )
		//		
		//		*info = -11;
		//    
		//    else if ( X->ncol < 0 || Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
		//	      B->ncol != X->ncol || X->Stype != SLU_DN ||
		//	      X->Dtype != SLU_D || X->Mtype != SLU_GE )
		//	
		//	*info = -12;
		//}
    }
    if (*info != 0) {
		i = -(*info);
		xerbla_("pdgssvx", &i);
		return (SPIKE_ERROR);
    }
    
    printf("colcnt_h[0] %lld\n", superlumt_options->colcnt_h[0]);

    /* ------------------------------------------------------------
       Allocate storage and initialize statistics variables. 
       ------------------------------------------------------------*/
    panel_size = superlumt_options->panel_size;
    relax = superlumt_options->relax;
    StatAlloc(n, nprocs, panel_size, relax, &Gstat);
    StatInit(n, nprocs, &Gstat);
    utime = Gstat.utime;
    ops = Gstat.ops;
    
    /* ------------------------------------------------------------
       Convert A to NC format when necessary.
       ------------------------------------------------------------*/
    if ( A->Stype == SLU_NR ) {
		NRformat *Astore = A->Store;
		AA = (SuperMatrix *) spike_malloc( ALIGN_INT, 1, sizeof(SuperMatrix));
		
		dCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
			       Astore->nzval, Astore->colind, Astore->rowptr,
			       SLU_NC, A->Dtype, A->Mtype);
	
		if ( notran ) { /* Reverse the transpose argument. */
	    	trant = TRANS;
	    	notran = 0;
		} else {
	    	trant = NOTRANS;
	    	notran = 1;
		}
    } else { /* A->Stype == NC */
		trant = superlumt_options->trans;
		AA = A;
    }

    /* ------------------------------------------------------------
       Diagonal scaling to equilibrate the matrix.
       ------------------------------------------------------------*/
    if ( equil ) {
		t0 = SuperLU_timer_();
		/* Compute row and column scalings to equilibrate the matrix A. */
		dgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);
	
		if ( info1 == 0 ) {
	    	/* Equilibrate matrix A. */
	    	dlaqgs(AA, R, C, rowcnd, colcnd, amax, equed);
	    	rowequ = (*equed == ROW) || (*equed == BOTH);
	    	colequ = (*equed == COL) || (*equed == BOTH);
		}
		utime[EQUIL] = SuperLU_timer_() - t0;
    }

    /* ------------------------------------------------------------
       Scale the right hand side.
       ------------------------------------------------------------*/
    
    /* ------------------------------------------------------------
       Perform the LU factorization.
       ------------------------------------------------------------*/
    if ( dofact || equil ) {

        /* Obtain column etree, the column count (colcnt_h) and supernode
	   partition (part_super_h) for the Householder matrix. */
    	t0 = SuperLU_timer_();
    	sp_colorder(AA, perm_c, superlumt_options, &AC);
    	utime[ETREE] = SuperLU_timer_() - t0;

 
    	printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
    		relax, panel_size, sp_ienv(3), sp_ienv(4));
    	fflush(stdout);


		/* Compute the LU factorization of A*Pc. */
    	t0 = SuperLU_timer_();
    	pdgstrf(superlumt_options, &AC, perm_r, L, U, &Gstat, info);
    	utime[FACT] = SuperLU_timer_() - t0;

    	flopcnt = 0;
    	for (i = 0; i < nprocs; ++i)flopcnt += Gstat.procstat[i].fcops;
    		ops[FACT] = flopcnt;

    	if ( superlumt_options->lwork == -1 ) {
    		superlu_memusage->total_needed = *info - A->ncol;
    		fprintf(stderr, "\n%s: Too much memory required!\n", __FUNCTION__ );
    		return (SPIKE_ERROR);
    	}
    }

    if ( *info > 0 ) {
    	if ( *info <= A->ncol ) {
	    /* Compute the reciprocal pivot growth factor of the leading
	       rank-deficient *info columns of A. */
    		*recip_pivot_growth = dPivotGrowth(*info, AA, perm_c, L, U);
    	}
    }
    else{
	/* ------------------------------------------------------------
	   Compute the reciprocal pivot growth factor *recip_pivot_growth.
	   ------------------------------------------------------------*/
	   *recip_pivot_growth = dPivotGrowth(A->ncol, AA, perm_c, L, U);

	/* ------------------------------------------------------------
	   Estimate the reciprocal of the condition number of A.
	   ------------------------------------------------------------*/
	   t0 = SuperLU_timer_();
	   if ( notran ) {
	   	*(unsigned char *)norm = '1';
	   } else {
	   	*(unsigned char *)norm = 'I';
	   }
	   anorm = dlangs(norm, AA);
	   dgscon(norm, L, U, anorm, rcond, info);
	   utime[RCOND] = SuperLU_timer_() - t0;


	/* ------------------------------------------------------------
	   Use iterative refinement to improve the computed solution and
	   compute error bounds and backward error estimates for it.
	   ------------------------------------------------------------*/

	/* ------------------------------------------------------------
	   Transform the solution matrix X to a solution of the original
	   system.
	   ------------------------------------------------------------*/

	/* Set INFO = A->ncol+1 if the matrix is singular to 
	   working precision.*/
	   		if ( *rcond < dlamch_("E") ) *info = A->ncol + 1;

	} /* end bracket of no error if-else check */

	superlu_dQuerySpace(nprocs, L, U, panel_size, superlu_memusage);

    /* ------------------------------------------------------------
       Deallocate storage after factorization.
       ------------------------------------------------------------*/
    if ( dofact || equil ) {
    	Destroy_CompCol_Permuted(&AC);
    }
    
    if ( A->Stype == SLU_NR ) {
		Destroy_SuperMatrix_Store(A);

		// Destroy_SuperMatrix_Store(AA);
		// SUPERLU_FREE(AA);
    }

    /* return AA instead of AA */
    A = AA;

    PrintStat(&Gstat);
    StatFree(&Gstat);
};


Error_t spike_pdgssvx_solve  (integer_t nprocs,
				superlumt_options_t *superlumt_options,
				SuperMatrix *AA, 
				integer_t *perm_c,
				integer_t *perm_r,
				equed_t *equed,
				double *R,
				double *C,
				SuperMatrix *L,
				SuperMatrix *U,
				SuperMatrix *B,
				SuperMatrix *X,
				double *recip_pivot_growth, 
				double *rcond,
				double *ferr,
				double *berr, 
				superlu_memusage_t *superlu_memusage,
				integer_t *info)
{
    DNformat  *Bstore, *Xstore;
    double    *Bmat, *Xmat;
    integer_t         n = AA->nrow;
    integer_t       nrhs = B->ncol;
    integer_t       ldb = n;
    integer_t       ldx = n;
    integer_t       colequ, equil, dofact, notran, rowequ;
    char      norm[1];
    trans_t   trant = TRANS; /* TODO fixed by me ! */
    integer_t     j, info1;
    int i;
    double amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
    integer_t       relax, panel_size;
    Gstat_t   Gstat;
    double    t0;      /* temporary time */
    double    *utime;
    flops_t   *ops, flopcnt;
   
    /* External functions */
    extern real_t 	dlangs(char *, SuperMatrix *);
    extern real_t 	dlamch_(char *);
    extern void 	sp_colorder(SuperMatrix *,
    				integer_t *,
    				superlumt_options_t *,
	    			SuperMatrix *);

    Bstore = B->Store;
    Xstore = X->Store;
    Bmat   = Bstore->nzval;
    Xmat   = Xstore->nzval;
    ldb    = Bstore->lda;
    ldx    = Xstore->lda;
    superlumt_options->perm_c = perm_c;
    superlumt_options->perm_r = perm_r;

    *info = 0;

    dofact = (superlumt_options->fact == DOFACT);
    equil  = (superlumt_options->fact == EQUILIBRATE);
    notran = (superlumt_options->trans == NOTRANS);
    
    if (dofact || equil) {
		*equed = NOEQUIL;
		rowequ = FALSE;
		colequ = FALSE;
    }
    else {
		rowequ = (*equed == ROW) || (*equed == BOTH);
		colequ = (*equed == COL) || (*equed == BOTH);
		smlnum = dlamch_("Safe minimum");
		bignum = 1. / smlnum;
    }

    /* ------------------------------------------------------------
       Test the input parameters.
       ------------------------------------------------------------*/

    /* ------------------------------------------------------------
       Convert A to NC format when necessary.
       ------------------------------------------------------------*/
    if ( AA->Stype == SLU_NR ) {
		fprintf(stderr, "\nVamos mal....");
    } else { /* A->Stype == NC */
		trant = superlumt_options->trans;
    }

    fprintf(stderr, "\n%s: line %d", __FUNCTION__, __LINE__ );

    /* ------------------------------------------------------------
       Allocate storage and initialize statistics variables. 
       ------------------------------------------------------------*/
    panel_size = superlumt_options->panel_size;
    relax      = superlumt_options->relax;
    
    StatAlloc(n, nprocs, panel_size, relax, &Gstat);
    StatInit(n, nprocs, &Gstat);

    fprintf(stderr, "\n%s: line %d", __FUNCTION__, __LINE__ );

    
    utime = Gstat.utime;
    ops = Gstat.ops;

    /* ------------------------------------------------------------
       Scale the right hand side.
       ------------------------------------------------------------*/
    if ( notran ) {
		if ( rowequ ) {
	    	for (j = 0; j < nrhs; ++j)
				for (i = 0; i < n; ++i) {
                    Bmat[i + j*ldb] *= R[i];
				}
		}
    }
    else if ( colequ ) {
		for (j = 0; j < nrhs; ++j)
	    	for (i = 0; i < n; ++i) {
                Bmat[i + j*ldb] *= C[i];
	    	}
    }
    
	/* ------------------------------------------------------------
	   Compute the solution matrix X.
	   ------------------------------------------------------------*/
	for (j = 0; j < nrhs; j++)    /* Save a copy of the right hand sides */
	    for (i = 0; i < n; i++)
			Xmat[i + j*ldx] = Bmat[i + j*ldb];

    fprintf(stderr, "\n%s: line %d", __FUNCTION__, __LINE__ );

    
	t0 = SuperLU_timer_();
	dgstrs(trant, L, U, perm_r, perm_c, X, &Gstat, info);
	utime[SOLVE] = SuperLU_timer_() - t0;
	ops[SOLVE] = ops[TRISOLVE];


	for(int i=0; i < 10; i++){
		fprintf(stderr, "\n x[%d] %f b[%d] %f", i, Xmat[i], i, Bmat[i] );
	}
    
	/* ------------------------------------------------------------
	   Use iterative refinement to improve the computed solution and
	   compute error bounds and backward error estimates for it.
	   ------------------------------------------------------------*/
	t0 = SuperLU_timer_();
	dgsrfs(trant, AA, L, U, perm_r, perm_c, *equed, R, C, B, X, ferr, berr, &Gstat, info);
	utime[REFINE] = SuperLU_timer_() - t0;

	/* ------------------------------------------------------------
	   Transform the solution matrix X to a solution of the original
	   system.
	   ------------------------------------------------------------*/
	if ( notran ) {
	    if ( colequ ) {
			for (j = 0; j < nrhs; ++j)
		    	for (i = 0; i < n; ++i) {
		    		Xmat[i + j*ldx] *= C[i];
		    	}
	    	}
	}
	else if ( rowequ ) {
	    for (j = 0; j < nrhs; ++j)
			for (i = 0; i < n; ++i) {
				Xmat[i + j*ldx] *= R[i];
			}
	}

    superlu_dQuerySpace(nprocs, L, U, panel_size, superlu_memusage);

    /* ------------------------------------------------------------
       Print timings, then deallocate statistic variables.
       ------------------------------------------------------------*/
    PrintStat(&Gstat);
    StatFree(&Gstat);

    return (SPIKE_SUCCESS);
};




Error_t solver_Configure( DirectSolverHander_t *handler )
{	
	/* create superlu structures from data */
	handler->nprocs     = (integer_t) 1;
    handler->fact       = (fact_t)   EQUILIBRATE;
    handler->trans      = (trans_t)  TRANS;
    handler->refact     = (yes_no_t) NO;
    handler->usepr      = (yes_no_t) NO;
    handler->equed      = (equed_t)  NOEQUIL;
    handler->info       = 0;
    handler->lwork      = 0;
    handler->panel_size = sp_ienv(1);
    handler->relax      = sp_ienv(2);
    handler->u          = 1.0;
    handler->drop_tol   = 0.0;
    handler->rpg        = 0.0;
    handler->rcond      = 0.0;
    handler->permc_spec = 1;
    handler->recip_pivot_growth = 0.0; // ?? 

    // handler->Gstat;


    /* allocate working buffer if it is not provided */
    if ( handler->lwork > 0 ) {
      handler->work = SUPERLU_MALLOC(handler->lwork);
      fprintf(stderr, "\nUse work space of size LWORK = " IFMT " bytes\n", handler->lwork);
  
      if ( handler->work = NULL ){ 
        	fprintf(stderr, "\n%s: Error %d\n", __FUNCTION__, __LINE__ );
        	SUPERLU_ABORT("SLINSOLX: cannot allocate work[]");
    	}
    }

    /* resume and exit */
    return (SPIKE_SUCCESS);
};


Error_t solver_Factorize(DirectSolverHander_t *handler,
						const integer_t n,
						const integer_t nnz,
						integer_t *restrict colind,
						integer_t *restrict rowptr,
						complex_t *restrict aij,
						complex_t *restrict xij,
						complex_t *restrict bij)
{
	/* set matrix dimensions */

	if ( n <= 0 || nnz <= 0 ){
		fprintf(stderr, "\n%s: Invalid number of rows (%d) or nnz (%d) elements in the matrix A", __FUNCTION__, n, nnz );
		abort();
	}

    handler->n          = n;
    handler->ldx        = n;
    handler->nnz        = nnz;

	/* create SuperLU matrix using handler.A pointer*/
	// dCreate_CompCol_Matrix( &handler->A, handler->n, handler->n, handler->nnz, aij, colind, rowptr, SLU_NR, SLU_D, SLU_GE);


	/* ya que la matriz esta ordenada como SLU_NR, la cambiamos a AA */
// dCreate_CompCol_Matrix

	dCreate_CompCol_Matrix( &handler->A, handler->n, handler->n, handler->nnz, 
			       aij, colind, rowptr,
			       SLU_NC, SLU_D, SLU_GE );


	/* row and column permutation arrays */
	handler->perm_r = (integer_t*) spike_malloc( ALIGN_INT , handler->n   , sizeof(integer_t));
	handler->perm_c = (integer_t*) spike_malloc( ALIGN_INT , handler->n   , sizeof(integer_t));

	/* row and column scaling vectors */
	handler->C      = (real_t*   ) spike_malloc( ALIGN_REAL, handler->n   , sizeof(real_t)   );
	handler->R      = (real_t*   ) spike_malloc( ALIGN_REAL, handler->n   , sizeof(real_t)   );

	/* get permutation spect */
	get_perm_c( handler->permc_spec, &handler->A, handler->perm_c);

	/* allocate space for elimination tree */
    handler->etree        = (integer_t*) spike_malloc( ALIGN_INT, handler->n, sizeof(integer_t));
    handler->colcnt_h     = (integer_t*) spike_malloc( ALIGN_INT, handler->n, sizeof(integer_t));
    handler->part_super_h = (integer_t*) spike_malloc( ALIGN_INT, handler->n, sizeof(integer_t));


	/* set superlu options */
    handler->superlumt_options.nprocs            = handler->nprocs;
    handler->superlumt_options.fact              = handler->fact;
    handler->superlumt_options.trans             = handler->trans;
    handler->superlumt_options.refact            = handler->refact;
    handler->superlumt_options.panel_size        = handler->panel_size;
    handler->superlumt_options.relax             = handler->relax;
    handler->superlumt_options.usepr             = handler->usepr;
    handler->superlumt_options.drop_tol          = handler->drop_tol;
    handler->superlumt_options.diag_pivot_thresh = handler->u;
    handler->superlumt_options.SymmetricMode     = NO;
    handler->superlumt_options.PrintStat         = YES;
    handler->superlumt_options.perm_c            = handler->perm_c;
    handler->superlumt_options.perm_r            = handler->perm_r;
    handler->superlumt_options.work              = handler->work;
    handler->superlumt_options.lwork             = handler->lwork;
	handler->superlumt_options.etree             = handler->etree;
	handler->superlumt_options.colcnt_h          = handler->colcnt_h;
	handler->superlumt_options.part_super_h      = handler->part_super_h;


	/* perform the factorization of the coefficient matrix */
	/* using the modified SuperLU's p?gssv expert driver   */
	spike_pdgssvx_factor( handler->nprocs,
		&handler->superlumt_options,
		&handler->A, 
		handler->perm_c,
		handler->perm_r,
		&handler->equed,
		handler->R,
		handler->C,
		&handler->L,
		&handler->U,
		&handler->recip_pivot_growth, 
		&handler->rcond,
		&handler->superlu_memusage,
		&handler->info);

	fprintf(stderr, "Fin de la rutina!!!\n");

	/* resume */
	return (SPIKE_SUCCESS);
};

Error_t solver_ApplyToRHS ( DirectSolverHander_t* handler,
							const integer_t nrhs,
							complex_t *restrict xij,
							complex_t *restrict bij)
{
	/* update number of RHS */
	handler->nrhs = nrhs;


	/* Create RHS and solution blocks */
    dCreate_Dense_Matrix( &handler->B, handler->n, handler->nrhs, bij, handler->n, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix( &handler->X, handler->n, handler->nrhs, xij, handler->n, SLU_DN, SLU_D, SLU_GE);


    // TODO usar las funciones de FILL RHS e iniciar estos vectores en el init()
    // de esta forma no tendremos que crear las estructuras cada vez


	/* backward and forward error propagation arrays */
	handler->berr   = (real_t*   ) spike_malloc( ALIGN_REAL, handler->nrhs, sizeof(real_t)   ); 
	handler->ferr   = (real_t*   ) spike_malloc( ALIGN_REAL, handler->nrhs, sizeof(real_t)   );

	/* perform the factorization of the coefficient matrix */
	/* using the modified SuperLU's p?gssv expert driver   */
	spike_pdgssvx_solve(
		handler->nprocs,
		&handler->superlumt_options,
		&handler->A, 
		handler->perm_c,
		handler->perm_r,
		&handler->equed,
		handler->R,
		handler->C,
		&handler->L, 
		&handler->U,
		&handler->B, 
		&handler->X,
		&handler->recip_pivot_growth, 
		&handler->rcond,
		handler->ferr,
		handler->berr, 
		&handler->superlu_memusage,
		&handler->info);

	for(int i=0; i < 10; i++)
		fprintf(stderr, "\nx %f b %f", xij[i], bij[i] );

	return (SPIKE_SUCCESS);
};

Error_t solver_Finalize( DirectSolverHander_t *handler )
{
	spike_nullify( handler->perm_c );
	spike_nullify( handler->perm_r );
	spike_nullify( handler->berr   );
	spike_nullify( handler->ferr   );
	spike_nullify( handler->C      );
	spike_nullify( handler->R      );
	spike_nullify( handler->etree  );
	spike_nullify( handler->colcnt_h);
	spike_nullify( handler->part_super_h);
	
	/* Destroy just the Store pointer of the SuperMatrix structure */
	/* Destroy_CompCol_Matrix( &handler->A ); deallocates the      */
	/* whole structure and gives more frees() than mallocs() when  */
	/* using valgrind                                              */
	Destroy_SuperMatrix_Store( &handler->A );
	Destroy_SuperMatrix_Store( &handler->B );
	Destroy_SuperMatrix_Store( &handler->X );

	// Destroy_SuperNode_SCP( &handler->U );
	// StatFree(&handler->Gstat);

   if ( handler->lwork == 0 ) {
        Destroy_SuperNode_SCP( &handler->L);
        Destroy_CompCol_NCP( &handler->U);
    }
    else{
    	spike_nullify( handler->work );
    }

	spike_nullify(handler);

	return (SPIKE_SUCCESS);
};


int main(int argc, const char *argv[])
{
	fprintf(stderr, "\nInner direct solver test.\n");

	/* -------------------------------------------------------------------- */
	/* .. Local variables. */
	/* -------------------------------------------------------------------- */
	integer_t 	nrhs;
	Error_t 	error;


	/* -------------------------------------------------------------------- */
	/* .. Load reference matrix from file. */
	/* -------------------------------------------------------------------- */
	matrix_t* A = matrix_LoadCSR("../Tests/pentadiagonal/small.bin");
	matrix_PrintAsDense(A, "Test matrix");
	
	/* -------------------------------------------------------------------- */
	/* .. CASE 1 Single RHS case */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 1: solve a simple system with a single"
					" vector on the RHS\n");

	nrhs         = 1;
	block_t*  x1 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b1 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x1, __zero  ); 
	block_InitializeToValue( b1, __punit ); 

	/* solve the linear system using the highest level call */
	superlu_solve( A->n, A->nnz, nrhs, A->colind, A->rowptr, A->aij , x1->aij, b1->aij);

	/* show the solution of the system */
	block_Print( x1, "Solution of the linear system");

	/* check the residual */
	// ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x1->aij, b1->aij, A->n, b1->m );
	
	block_Deallocate ( x1 );
	block_Deallocate ( b1 );

	/* -------------------------------------------------------------------- */
	/* .. CASE 2 Solve the system for multiple RHSs.                        */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 2: solve a simple system with two vectors"
		            " on a single RHS\n");

	nrhs         = 2;
	block_t*  x2 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b2 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x2, __zero  ); 
	block_InitializeToValue( b2, __punit );

	for(integer_t i=0; i < (A->n * nrhs); i++ )
		b2->aij[i] = (i < A->n ) ? __punit : __nunit;

	/* solve the linear system using the highest level call */
	// superlu_solve( A->n, A->nnz, nrhs, A->colind, A->rowptr, A->aij , x2->aij, b2->aij);

	/* show residual */
	block_Print( x2, "Solutiom of the linear system");

	/* check the residual */
	// ComputeResidualOfLinearSystem( A->colind, A->rowptr, A->aij, x2->aij, b2->aij, A->n, b2->m );

	
	/* clean up and resume */
	block_Deallocate ( x2 );
	block_Deallocate ( b2 );	
	

	/* -------------------------------------------------------------------- */
	/* .. CASE 4 Factorizes the matrix and then solves for multiple RHS   . */
	/* ..        separately                                               . */
	/* -------------------------------------------------------------------- */
	fprintf(stderr, "\nTEST CASE 2: solve a simple system with two vectors"
	            " on a single RHS\n");
	
	nrhs         = 2;
	block_t*  x3 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_t*  b3 = block_CreateEmptyBlock( A->n, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	block_InitializeToValue( x3, __zero  ); 
	block_InitializeToValue( b3, __punit );

	for(integer_t i=0; i < (A->n * nrhs); i++ ) b3->aij[i] = (i < A->n ) ? __punit : __nunit;

	matrix_PrintAsDense(A, "Original matrix");

	DirectSolverHander_t *handler = solver_Init();

	solver_Configure( handler );

	solver_Factorize( handler, A->n, A->nnz, A->colind, A->rowptr, A->aij, x3->aij, b3->aij );

	fprintf(stderr, "Todo bien %d\n", __LINE__);

	solver_ApplyToRHS( handler, nrhs, x3->aij, b3->aij);
	fprintf(stderr, "Todo bien %d\n", __LINE__);

	solver_Finalize ( handler );
	fprintf(stderr, "Todo bien %d\n", __LINE__);

	block_Deallocate( x3 );
	fprintf(stderr, "Todo bien %d\n", __LINE__);

	block_Deallocate( b3 );
	fprintf(stderr, "Todo bien %d\n", __LINE__);


	/* -------------------------------------------------------------------- */
	/* .. Clean up and resume                                             . */
	/* -------------------------------------------------------------------- */

	matrix_Deallocate( A );
	fprintf(stderr, "\nTest finished finished\n");

	return 0;
}
