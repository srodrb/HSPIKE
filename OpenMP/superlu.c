
#include "pdsp_defs.h"
#include "util.h"

#include "spike_algebra.h"

/*
 * Decomposes the matrix into L and U factors and then
 * uses them to solve various linear systems.
 */

void superlu_solve (const integer_t n, 
                    const integer_t nnz,
                    const integer_t nrhs,
                    integer_t *restrict colind,
                    integer_t *restrict rowptr,
                    complex_t *restrict aij )
{
  /* asub = colind and rowptr is rowptr */

    SuperMatrix              A, L, U;
    SuperMatrix              B, X;
    NCformat                *Astore;
    SCPformat               *Lstore;
    NCPformat               *Ustore;
    complex_t               *rhsb, *rhsx, *xact;

    /*
     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = 0: natural ordering 
     *   permc_spec = 1: minimum degree ordering on structure of A'*A
     *   permc_spec = 2: minimum degree ordering on structure of A'+A
     *   permc_spec = 3: approximate minimum degree for unsymmetric matrices
     */
     integer_t permc_spec = 3;

    /* row and column permutation vectors */
    integer_t *perm_c = (integer_t*) spike_malloc( ALIGN_INT, n, sizeof(integer_t)); /* column permutation vector */
    integer_t *perm_r = (integer_t*) spike_malloc( ALIGN_INT, n, sizeof(integer_t)); /* row permutations from partial pivoting */
    
    /* Extra variables */
    superlumt_options_t      superlumt_options;
    superlu_memusage_t       superlu_memusage;
    
    /* Default parameters to control factorization. */
    integer_t                nprocs = 1;
    fact_t                   fact = EQUILIBRATE;
    trans_t                  trans = NOTRANS;
    yes_no_t                 refact = NO;
    yes_no_t                 usepr  = NO;
    equed_t                  equed = NOEQUIL;
    void                    *work;
    integer_t                info;
    integer_t                ldx = n;
    integet_t                lwork = 0;
    integet_t                panel_size = sp_ienv(1);
    integer_t                relax = sp_ienv(2);
    integer_t                i;
    complex_t               *R, *C;
    complex_t               *ferr, *berr;
    complex_t                u = 1.0;
    real_t                   drop_tol = 0.0;
    real_t                   rpg;
    real_t                   rcond;

    if ( lwork > 0 ) {
      work = SUPERLU_MALLOC(lwork);
      fprintf(stderr, "\nUse work space of size LWORK = " IFMT " bytes\n", lwork);
  
      if ( !work ) 
        SUPERLU_ABORT("SLINSOLX: cannot allocate work[]");
    }

    /* case set up */
    get_perm_c(permc_spec, &A, perm_c);

    superlumt_options.nprocs            = nprocs;
    superlumt_options.fact              = fact;
    superlumt_options.trans             = trans;
    superlumt_options.refact            = refact;
    superlumt_options.panel_size        = panel_size;
    superlumt_options.relax             = relax;
    superlumt_options.usepr             = usepr;
    superlumt_options.drop_tol          = drop_tol;
    superlumt_options.diag_pivot_thresh = u;
    superlumt_options.SymmetricMode     = NO;
    superlumt_options.PrintStat         = NO;
    superlumt_options.perm_c            = perm_c;
    superlumt_options.perm_r            = perm_r;
    superlumt_options.work              = work;
    superlumt_options.lwork             = lwork;
    
    if ( !(superlumt_options.etree = intMalloc(n)) )
      SUPERLU_ABORT("Malloc fails for etree[].");
    if ( !(superlumt_options.colcnt_h = intMalloc(n)) )
      SUPERLU_ABORT("Malloc fails for colcnt_h[].");
    if ( !(superlumt_options.part_super_h = intMalloc(n)) )
      SUPERLU_ABORT("Malloc fails for colcnt_h[].");

    dCreate_CompCol_Matrix(&A, m, n, nnz, aij, colind, rowptr, SLU_NC, SLU_S, SLU_GE);
    Astore = A.Store;
    fprintf(stderr, "Dimension " IFMT "x" IFMT "; # nonzeros " IFMT "\n", A.nrow, A.ncol, Astore->nnz);

    /* create RHS matrix structures from input data */
    dCreate_Dense_Matrix(&B, m, nrhs, b, m, SLU_DN, SLU_S, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, x, m, SLU_DN, SLU_S, SLU_GE);


    if (!(R = (complex_t*) SUPERLU_MALLOC(A.nrow * sizeof(complex_t)))) 
        SUPERLU_ABORT("SUPERLU_MALLOC fails for R[].");
    if ( !(C = (complex_t *) SUPERLU_MALLOC(A.ncol * sizeof(complex_t))) )
        SUPERLU_ABORT("SUPERLU_MALLOC fails for C[].");
    if ( !(ferr = (complex_t *) SUPERLU_MALLOC(nrhs * sizeof(complex_t))) )
        SUPERLU_ABORT("SUPERLU_MALLOC fails for ferr[].");
    if ( !(berr = (complex_t *) SUPERLU_MALLOC(nrhs * sizeof(complex_t))) ) 
        SUPERLU_ABORT("SUPERLU_MALLOC fails for berr[].");
    
    /* ------------------------------------------------------------
       WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: AX = B
       ------------------------------------------------------------*/
    psgssvx(nprocs, &superlumt_options, &A, perm_c, perm_r,
      &equed, R, C, &L, &U, &B, &X, &rpg, &rcond,
      ferr, berr, &superlu_memusage, &info);

    if ( info == 0 || info == n+1 ) {
      printf("Recip. pivot growth = %e\n", rpg);
      printf("Recip. condition number = %e\n", rcond);
      printf("%8s%16s%16s\n", "rhs", "FERR", "BERR");
      
      for (i = 0; i < nrhs; ++i)
        printf(IFMT "%16e%16e\n", i+1, ferr[i], berr[i]);
           
      Lstore = (SCPformat *) L.Store;
      Ustore = (NCPformat *) U.Store;
      
      printf("No of nonzeros in factor L = " IFMT "\n", Lstore->nnz);
      printf("No of nonzeros in factor U = " IFMT "\n", Ustore->nnz);
      printf("No of nonzeros in L+U = " IFMT "\n", Lstore->nnz + Ustore->nnz - n);
      printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions " IFMT "\n",
      
      superlu_memusage.for_lu/1e6, superlu_memusage.total_needed/1e6,
      superlu_memusage.expansions);
         
      fflush(stdout);
    } else if ( info > 0 && lwork == -1 ) { 
      printf("** Estimated memory: " IFMT " bytes\n", info - n);
    }

    printf("First system: psgssvx(): info " IFMT "\n----\n", info);


    /* clean up and resume execution */
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);


    spike_nullify (rhsb);
    spike_nullify (rhsx);
    spike_nullify (xact);
    spike_nullify (perm_r);
    spike_nullify (perm_c);
    spike_nullify (R);
    spike_nullify (C);
    spike_nullify (ferr);
    spike_nullify (berr);
    Destroy_CompCol_Matrix(&A1);
    Destroy_SuperMatrix_Store(&B1);
    Destroy_SuperMatrix_Store(&X);
    spike_nullify (superlumt_options.etree);
    spike_nullify (superlumt_options.colcnt_h);
    spike_nullify (superlumt_options.part_super_h);

    if ( lwork == 0 ) {
        Destroy_SuperNode_SCP(&L);
        Destroy_CompCol_NCP(&U);
    } else if ( lwork > 0 ) {
        spike_nullify(work);
    }

    fprintf(stderr, "\nFin del solver SUPERLU\n");
};
