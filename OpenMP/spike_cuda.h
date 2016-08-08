/*
 * =====================================================================================
 *
 *       Filename:  spike_cuda.h
 *
 *    Description:  Linear algebra backed based on NVidia cuSolver and cuSparse libraries.
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:32:39
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * TODO: check CUDA version, should be >= 7.5 for optimum performance.
 * TODO: use low-level API calls for reordering (see Low Level Function Reference)
 *
 * =====================================================================================
 */
#ifndef _SPIKE_CUDA_H_
    #define _SPIKE_CUDA_H_

    #include "spike_datatypes.h"
    #include "spike_common.h"
    #include "spike_memory.h"

    #define ASYNC_COPY 1
    #define BLOCKING_GPU 1

    /* local alloca and free counters for device memory */
    extern unsigned int cnt_devMalloc;
    extern unsigned int cnt_devFree;

    /*
        According to the cuSolver API (section 1.4), these datatypes
        are used to work with complex numbers.
     */
    #if defined (_DATATYPE_Z_) // double complex
        /* complex numbers definition */
        #define  SPIKE_CUDA_PREC     Z
     
    #elif defined (_DATATYPE_C_) // complex float
        /* complex numbers definition */
        #define  SPIKE_CUDA_PREC     C
    
    #elif defined (_DATATYPE_D_) // double precision float
        #define  SPIKE_CUDA_PREC     D
    
    #else // single precision float
        #define  SPIKE_CUDA_PREC     S
    #endif

    /* cusolver and Cuda 7.5 preview (low-level API) */
    #include "cusparse.h"
    #include "cusolverSp.h"
    #include "cusolverSp_LOWLEVEL_PREVIEW.h"


    typedef struct {

        integer_t n;       /* coefficient matrix leading dimension      */
        integer_t nnz;     /* number of non zero elements in the matrix */
        integer_t nrhs;    /* number of right hand side columns         */

        /* device pointers */
        integer_t *d_colind; /* device colind pointer */
        integer_t *d_rowptr; /* device rowptr pointer */
        complex_t *d_aij;    /* device aij    pointer */
        complex_t *d_xij;    /* device xij pointer */
        complex_t *d_bij;    /* device bij pointer */

        /* host pointers */
        integer_t *h_colind; /* host colind pointer */
        integer_t *h_rowptr; /* host rowptr pointer */
        complex_t *h_aij;    /* host aij    pointer */
        complex_t *h_xij;    /* host xij pointer */
        complex_t *h_bij;    /* host bij pointer */

        /* Current CUDA stream */
        cudaStream_t stream[10];

        /* Flag to know whether the device memory for RHS has been allocated or not */
        Bool_t deviceMemForRHSisAllocated;
        integer_t deviceRHSBlockDist;

        /* Max device memory and current memory usage */
        size_t deviceTotalMemory;
        size_t deviceFreeMemory;

        /* number of rhs fittin on the device after factorization */
        integer_t nrhs_fiffing;
        integer_t max_nrhs;
        integer_t nstreams;

        /* needed handlers */
        cusolverSpHandle_t cusolverHandle; /* cusolver handler */
        cusparseMatDescr_t MatDescr; /* cuSparse matrix descriptor */

        /* Low-level cusolver API structures */
        csrqrInfo_t csrqrInfo;

        /* Other local variables */
        integer_t issym;            /* 1 if so, 0 otherwise                       */   
        size_t internalDataInBytes; /* space for H matrix                         */
        size_t workspaceInBytes;    /* space for QR factorization                 */
        void   *d_work;             /* device workspace of workspaceInBytes bytes */



        /* -------------------------------------------------------------------- */
        /* .. Statistical variables                                             */
        /* -------------------------------------------------------------------- */
        spike_timer_t transfer_t;
        spike_timer_t ordering_t;
        spike_timer_t factor_t;
        spike_timer_t solve_t;
        spike_timer_t clean_t;
        
        integer_t rhs_block_count;
        integer_t rhs_column_count;

    } DirectSolverHander_t;

    DirectSolverHander_t *directSolver_CreateHandler(void);

    Error_t directSolver_Configure( DirectSolverHander_t *handler, const integer_t max_nrhs );


    Error_t directSolver_Factorize(DirectSolverHander_t *handler,
    						const integer_t n,
    						const integer_t nnz,
    						integer_t *__restrict__ colind,
    						integer_t *__restrict__ rowptr,
    						complex_t *__restrict__ aij);

    Error_t directSolver_SolveForRHS ( DirectSolverHander_t* handler,
                                const integer_t nrhs,
                                complex_t *__restrict__ xij,
                                complex_t *__restrict__ bij);


    Error_t directSolver_SolveForRHS_sync ( DirectSolverHander_t* handler,
                                const integer_t nrhs,
                                complex_t *__restrict__ xij,
                                complex_t *__restrict__ bij);

    Error_t directSolver_SolveForRHS_async_1 ( DirectSolverHander_t* handler,
                                const integer_t nrhs,
                                complex_t *__restrict__ xij,
                                complex_t *__restrict__ bij);


    Error_t directSolver_SolveForRHS_asyn_2 ( DirectSolverHander_t* handler,
                                const integer_t nrhs,
                                complex_t *__restrict__ xij,
                                complex_t *__restrict__ bij);


    Error_t directSolver_Finalize( DirectSolverHander_t *handler );

    Error_t directSolver_ShowStatistics( DirectSolverHander_t *handler );

    Error_t directSolver_Solve (integer_t n,
                                integer_t nnz,
                                integer_t nrhs,
                                integer_t *__restrict__ colind,
                                integer_t *__restrict__ rowptr,
                                complex_t *__restrict__ aij,
                                complex_t *__restrict__ x,
                                complex_t *__restrict__ b);

    Error_t cuda_standalone_symrcm(integer_t n,
                                integer_t nnz,
                                integer_t *__restrict__ rowptr,
                                integer_t *__restrict__ colind);

#endif /* end of _SPIKE_CUDA_H_ definition */