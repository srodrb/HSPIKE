#include "spike_cuda.h"

unsigned int cnt_devMalloc = 0;
unsigned int cnt_devFree   = 0;

static inline void checkCudaErrors( int status ) { /* TODO */ };

static inline void spike_devMalloc( void* devPtr, const size_t nmemb, const size_t size )
{
	checkCudaErrors( cudaMalloc((void**) &devPtr, nmemb * size ));

	cnt_devMalloc += 1;
};

static inline Error_t spike_devNullify( void* devPtr )
{
	if( devPtr) { checkCudaErrors( cudaFree( devPtr )); }

	cnt_devFree += 1;

	return (SPIKE_SUCCESS);
};


DirectSolverHander_t *directSolver_CreateHandler(void)
{

	// TODO check CUDA version >= 7.5

	DirectSolverHander_t *handler = (DirectSolverHander_t*) spike_malloc( ALIGN_INT, 1, sizeof(DirectSolverHander_t));

	return (handler);
};

Error_t directSolver_Configure( DirectSolverHander_t *handler )
{
	/* -------------------------------------------------------------------- */
	/* .. Query used and free memory on the device. */
	/* -------------------------------------------------------------------- */	
	checkCudaErrors( cudaMemGetInfo( &handler->freeMem, &handler->usedMem ));

	/* -------------------------------------------------------------------- */
	/* .. Initilialize handlers.                                            */
	/* -------------------------------------------------------------------- */
	checkCudaErrors( cusolverSpCreate ( &handler->cusolverHandle));
	checkCudaErrors( cusolverSpCreateCsrqrInfo ( &handler->csrqrInfo ));

	/* -------------------------------------------------------------------- */
	/* .. Setup matrix information.                                         */
	/* -------------------------------------------------------------------- */
	checkCudaErrors( cusparseSetMatType      ( handler->MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	checkCudaErrors( cusparseSetMatIndexBase ( handler->MatDescr, CUSPARSE_INDEX_BASE_ZERO     ) );

	/* statistical parameters */
	handler->transfer_t = 0.0;
	handler->ordering_t = 0.0;
	handler->factor_t   = 0.0;
	handler->solve_t    = 0.0;
	handler->clean_t    = 0.0;

	handler->rhs_block_count  = 0;
	handler->rhs_column_count = 0;

	return (SPIKE_SUCCESS);
};


Error_t directSolver_Factorize(DirectSolverHander_t *handler,
						const integer_t n,
						const integer_t nnz,
						integer_t *__restrict__ colind,
						integer_t *__restrict__ rowptr,
						complex_t *__restrict__ aij)
{
	/* matrix dimensions */
	handler->n   = n;
	handler->nnz = nnz;

	/* -------------------------------------------------------------------- */
	/* .. Set pointers for device and host arrays                           */
	/* -------------------------------------------------------------------- */
	handler->h_aij    = aij;
	handler->h_colind = colind;
	handler->h_rowptr = rowptr;

	/* allocate memory for coefficient matrix on the device */
	spike_devMalloc( handler->d_aij   , handler->nnz  , sizeof(complex_t) );
	spike_devMalloc( handler->d_colind, handler->nnz  , sizeof(integer_t) );
	spike_devMalloc( handler->d_rowptr, handler->n + 1, sizeof(integer_t) );

	/* transfer the arrays to the device memory */
	checkCudaErrors( cudaMemcpy( handler->d_aij   , handler->h_aij   ,  handler->nnz    * sizeof(complex_t), cudaMemcpyHostToDevice ));
	checkCudaErrors( cudaMemcpy( handler->d_colind, handler->h_colind,  handler->nnz    * sizeof(integer_t), cudaMemcpyHostToDevice ));
	checkCudaErrors( cudaMemcpy( handler->d_rowptr, handler->h_rowptr, (handler->n + 1) * sizeof(integer_t), cudaMemcpyHostToDevice ));

	/* -------------------------------------------------------------------- */
	/* .. Verify if A is symmetric.                                         */
	/* -------------------------------------------------------------------- */
    checkCudaErrors(cusolverSpXcsrissymHost( handler->cusolverHandle,
    						handler->n,
    						handler->nnz,
    						handler->MatDescr,
    						handler->h_rowptr,
    						handler->h_rowptr +1,
    						handler->h_colind,
    						&handler->issym));
	/* -------------------------------------------------------------------- */
	/* .. Reorder to reduce fill-in.                                        */
	/*                                                                      */
	/* The low-level API does not reorder under the hood, so it is needed   */
	/* to do so explicitely.                                                */
	/* -------------------------------------------------------------------- */   

    // TODO.


	/* analyses sparsity pattern of H and Q matrices */
	checkCudaErrors( cusolverSpXcsrqrAnalysis ( handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           handler->MatDescr,
                           handler->d_rowptr,
                           handler->d_colind,
                           handler->csrqrInfo ));

	/* After the analysis, the size of working space to perform QR factorization can be retrieved */
	checkCudaErrors( cusolverSpScsrqrBufferInfo ( handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           handler->MatDescr,
                           (const float*) handler->d_aij,
                           handler->d_rowptr,
                           handler->d_colind,
                           handler->csrqrInfo,
                           &handler->internalDataInBytes,
                           &handler->workspaceInBytes));

	/* allocate work space for factorization */
	checkCudaErrors( cudaMalloc((void**) &handler->d_work, handler->workspaceInBytes) );
	
	/* This function shifts diagonal of A by parameter mu such that we can factorize */
	/* For linear solver, the user just sets mu to zero.                             */ 
	/* For eigenvalue solver, mu can be a value of shift in inverse-power method.    */
	checkCudaErrors( cusolverSpScsrqrSetup ( handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           handler->MatDescr,
                           (const float*) handler->d_aij,
                           handler->d_rowptr,
                           handler->d_colind,
                           0,
                           handler->csrqrInfo ));

	/* Perform numerical factorization */
	/* If either x or b is nil, only factorization is done. The user needs cusolverSpXcsrqrSolve    */
	/* to find the least-square solution.                                                           */
	/* If both x and b are not nil, QR factorization and solve are combined together. b is over-    */
	/* written by c and x is the solution of least-square.                                          */
	/* pBuffer: buffer allocated by the user, the size is returned by cusolverSpXcsrqrBufferInfo(). */
	checkCudaErrors( cusolverSpScsrqrFactor ( handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           NULL, /* handler->d_bij, */
                           NULL, /* handler->d_xij, */
                           handler->csrqrInfo,
                           handler->d_work ));


	/* resume and return */
	return (SPIKE_SUCCESS);
};


Error_t directSolver_SolveForRHS ( DirectSolverHander_t* handler,
                            const integer_t nrhs,
                            complex_t *__restrict__ xij,
                            complex_t *__restrict__ bij)
{
	/* update the value of rhs columns */
	handler->nrhs = nrhs;

	/* update statistics, keep track of RHS */
	handler->rhs_block_count  += 1;
	handler->rhs_column_count += nrhs;

	/* set the pointers to the values of the RHS */
	handler->h_xij = xij;
	handler->h_bij = bij;

	checkCudaErrors( cudaMalloc((void**) &handler->d_xij, handler->n * handler->nrhs * sizeof(complex_t)));
	checkCudaErrors( cudaMalloc((void**) &handler->d_bij, handler->n * handler->nrhs * sizeof(complex_t)));

	/* allocate memory for rhs vectors and solution on the device */
	cudaMemcpy( handler->d_xij, handler->h_xij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyHostToDevice );
	cudaMemcpy( handler->d_bij, handler->h_bij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyHostToDevice );

	/* Forward and backward substitution */
	checkCudaErrors( cusolverSpScsrqrSolve ( handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           (float*) handler->d_bij,
                           (float*) handler->d_xij,
                           handler->csrqrInfo,
                           handler->d_work ));

	/* transfer the solution back to the host */
	cudaMemcpy( handler->d_xij, xij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyDeviceToHost );

	/* deallocate rhs vectors on the device */
	spike_devNullify( handler->d_xij );
	spike_devNullify( handler->d_bij );

	fprintf(stderr, "\n%s: %d WARNING! make sure you cant rehuse these buffers!", __FUNCTION__, __LINE__ );

	/* resume and return */
	return (SPIKE_SUCCESS);
};

Error_t directSolver_ShowStatistics( DirectSolverHander_t *handler )
{



	return (SPIKE_SUCCESS);
};

Error_t directSolver_Finalize( DirectSolverHander_t *handler )
{
	spike_devNullify( handler->d_work  );
	spike_devNullify( handler->d_colind);
	spike_devNullify( handler->d_rowptr);
	spike_devNullify( handler->d_aij   );
	// realloc is possible? -> spike_devNullify( handler->d_xij  );
	// realloc is possible? -> spike_devNullify( handler->d_bij  );

	checkCudaErrors( cusolverSpDestroyCsrqrInfo ( handler->csrqrInfo ));

	if ( handler->cusolverHandle ) { checkCudaErrors( cusolverSpDestroy      ( handler->cusolverHandle)); }
	if ( handler->cusparseHandle ) { checkCudaErrors( cusparseDestroy        ( handler->cusparseHandle)); }
	if ( handler->MatDescr       ) { checkCudaErrors( cusparseDestroyMatDescr( handler->MatDescr      )); }

	spike_nullify(handler);

	return (SPIKE_SUCCESS);
};

 Error_t directSolver_Solve (integer_t n,
 							integer_t nnz,
 							integer_t nrhs,
 							integer_t *__restrict__ colind, // ja
							integer_t *__restrict__ rowptr, // ia
							complex_t *__restrict__ aij,
							complex_t *__restrict__ x,
							complex_t *__restrict__ b)
{
	return (SPIKE_SUCCESS);
};