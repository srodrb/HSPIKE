/*
 * =====================================================================================
 *
 *       Filename:  spike_backend.h
 *
 *    Description:  Define a generic interface for multiple direct solvers back-ends.
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

#ifndef _SPIKE_SUPERLU_
    #define _SPIKE_SUPERLU_

    #include "spike_datatypes.h"
    #include "spike_common.h"
    #include "spike_memory.h"

    #if defined (_DATATYPE_Z_) // double complex
        #include "slu_mt_zdefs.h"
        #define  MATRIX_DTYPE     SLU_Z
     
    #elif defined (_DATATYPE_C_) // complex float
        #include "slu_mt_cdefs.h"
        #define  MATRIX_DTYPE     SLU_C
    
    #elif defined (_DATATYPE_D_) // double precision float
        #include "slu_mt_ddefs.h"
        #define  MATRIX_DTYPE     SLU_D
    
    #else // single precision float
        #include "slu_mt_sdefs.h"
        #define  MATRIX_DTYPE     SLU_S
    #endif

    /* superlu interface */
    #include "slu_mt_util.h"

    typedef struct {
        
        superlumt_options_t     superlumt_options;
        superlu_memusage_t      superlu_memusage;
        // Gstat_t  Gstat;
            
        integer_t nprocs;
        fact_t fact;
        trans_t trans;
        yes_no_t refact;
        yes_no_t usepr;
        equed_t equed;
        void *work;
        integer_t info;
        integer_t lwork;
        integer_t panel_size;
        integer_t relax;
        integer_t permc_spec;
        real_t u;
        real_t drop_tol;
        real_t rpg;
        real_t recip_pivot_growth; // ???
        real_t rcond;

        integer_t n;    /* matrix dimension                    */
        integer_t ldx;  /* matrix leading dimension            */
        integer_t nnz;  /* nnz elements in A                   */
        integer_t nrhs; /* number of columns of x and b arrays */
             

        SuperMatrix             A; 
        SuperMatrix             L;
        SuperMatrix             U;
        SuperMatrix             B;
        SuperMatrix             X;

        integer_t               *perm_c;
        integer_t               *perm_r;
        real_t                  *R;
        real_t                  *C;
        real_t                  *berr;
        real_t                  *ferr;

        integer_t *etree;           /* elimination tree */
        integer_t *colcnt_h;        /* column count */
        integer_t *part_super_h;    /* supernode partition for the Householder matrix */


        /* -------------------------------------------------------------------- */
        /* .. Statistical variables                                             */
        /* -------------------------------------------------------------------- */
        spike_timer_t ordering_t;
        spike_timer_t scaling_t;
        spike_timer_t factor_t;
        spike_timer_t solve_t;
        spike_timer_t refine_t;
        integer_t rhs_block_count;
        integer_t rhs_column_count;

    } DirectSolverHander_t;

    DirectSolverHander_t *directSolver_CreateHandler(void);

    Error_t directSolver_Configure( DirectSolverHander_t *handler );


    Error_t directSolver_Factorize(DirectSolverHander_t *handler,
    						const integer_t n,
    						const integer_t nnz,
    						integer_t *restrict colind,
    						integer_t *restrict rowptr,
    						complex_t *restrict aij);

    Error_t directSolver_SolveForRHS ( DirectSolverHander_t* handler,
                                const integer_t nrhs,
                                complex_t *restrict xij,
                                complex_t *restrict bij);

    Error_t directSolver_ShowStatistics( DirectSolverHander_t *handler );

    Error_t directSolver_Finalize( DirectSolverHander_t *handler );

    Error_t directSolver_Solve (integer_t n,
                                integer_t nnz,
                                integer_t nrhs,
                                integer_t *restrict colind, // ja
                                integer_t *restrict rowptr, // ia
                                complex_t *restrict aij,
                                complex_t *restrict xij,
                                complex_t *restrict bij);

#endif /* end of _SPIKE_BACKEND_ definition */