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
#ifndef _SPIKE_PARDISO_H_
    #define _SPIKE_PARDISO_H_

    #include "spike_datatypes.h"
    #include "spike_common.h"
    #include "spike_memory.h"

    /* INTEL MKL interface */
    #include "mkl_types.h"
    #include "mkl_spblas.h"
    #include "mkl.h"
    #include "mkl_cblas.h"

    /* Pardiso interface */
    #include "mkl_pardiso.h"

    /*
       Depending on the back-end, the nature of the coeffient matrix is specified
       differently. In general, each backend has a predefined list of integer values
       to identify each type of matrix.
       Here we support some of them.
     */
    #ifndef _COMPLEX_ARITHMETIC_
        #define MTYPE_STRUC_SYMM    1   /* Real and structurally symmetric          */
        #define MTYPE_POSDEF        2   /* Real and symmetric positive definite     */
        #define MTYPE_SYMM_INDEF   -2   /* Real and symmetric indefinite            */
        #define MTYPE_GEN_NOSYMM   11   /* Real and nonsymmetric matrix             */
    #else
        #define MTYPE_STRUC_SYMM    3   /* Complex and structurally symmetric       */
        #define MTYPE_HERM_POSDEF   4   /* Complex and Hermitian positive definite  */
        #define MTYPE_HERM_INDEF   -4   /* Complex and Hermitian indefinite         */
        #define MTYPE_SYMM          6   /* Complex and symmetric matrix             */
        #define MTYPE_GEN_NOSYMM   13   /* Complex and nonsymmetric matrix          */
    #endif


    typedef struct {
        integer_t n; /* matrix dimension */
        integer_t nnz; /* number of non zero elements in the matrix */
        integer_t nrhs;

        integer_t *colind;
        integer_t *rowptr;
        complex_t *aij;

        complex_t *xij;
        complex_t *bij;

        MKL_INT mtype;

        MKL_INT *conf[64];
        MKL_INT iparm[64];

        MKL_INT maxfct, mnum, error, msglvl;

        /* -------------------------------------------------------------------- */
        /* .. Statistical variables                                             */
        /* -------------------------------------------------------------------- */
        spike_timer_t ordering_t;
        spike_timer_t factor_t;
        spike_timer_t solve_t;
        spike_timer_t clean_t;
        
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

    Error_t directSolver_Finalize( DirectSolverHander_t *handler );

    Error_t directSolver_ShowStatistics( DirectSolverHander_t *handler );

    Error_t directSolver_Solve (integer_t n,
                                integer_t nnz,
                                integer_t nrhs,
                                integer_t *restrict colind, // ja
                                integer_t *restrict rowptr, // ia
                                complex_t *restrict aij,
                                complex_t *restrict x,
                                complex_t *restrict b);

#endif /* end of _SPIKE_PARDISO_H_ definition */