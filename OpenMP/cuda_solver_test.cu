#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "cusparse.h"
#include "cusolverSp.h"

/* Cuda 7.5 preview */
#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include "helper_cuda.h"
#include "helper_cusolver.h"

#define integer_t int
#define complex_t float
#define real_t float

#define SOLVE_ON_GPU 1
#define CUSOLVER_NO_REORDERING 0

int main(int argc, char const *argv[])
{
	/* definition of a small matrix */
	const integer_t n     = 4;
	const integer_t nnz   = 9;
	const integer_t nrhs  = 1;

	/* additional variables related with the solver */
	integer_t issym    = 0;
	real_t    epsilon  = 1e-5;
	integer_t singular = 0;
	integer_t reorder  = CUSOLVER_NO_REORDERING;

	/* cuSolver handler */
	cusolverSpHandle_t cusolverHandle = NULL;

	/* cusparse handler */
	cusparseHandle_t cusparseHandle = NULL;

	/* cuSparse matrix descriptor */
	cusparseMatDescr_t MatDescr = NULL;

	complex_t *d_aij;
	integer_t *d_colind;
	integer_t *d_rowptr;
	complex_t *d_xij;
	complex_t *d_bij;

	/* create a dummy csr matrix */
	complex_t aij   [9] = {2., 4., 6., 1., 2., 4., 5., 1., 3. };
	integer_t colind[9] = {0, 1, 2, 1, 2, 1, 2, 3, 3 };
	integer_t rowptr[5] = {0, 3, 5, 8, 9};

	/* define solution and rhs vectors */
	complex_t xij[4] = {0.};
	complex_t bij[4] = {1.};

	/* allocate space on the device memory */
	checkCudaErrors( cudaMalloc((void**) &d_aij   ,  nnz  * sizeof(complex_t)) );
	checkCudaErrors( cudaMalloc((void**) &d_colind,  nnz  * sizeof(integer_t)) );
	checkCudaErrors( cudaMalloc((void**) &d_rowptr, (n+1) * sizeof(integer_t)) );
	checkCudaErrors( cudaMalloc((void**) &d_xij, n * nrhs * sizeof(complex_t)) );
	checkCudaErrors( cudaMalloc((void**) &d_bij, n * nrhs * sizeof(complex_t)) );

	/* transfer arrays to the device memory */
	checkCudaErrors( cudaMemcpy( d_aij   , aij   ,  nnz     * sizeof(complex_t), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_colind, colind,  nnz     * sizeof(integer_t), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_rowptr, rowptr, (n+1)    * sizeof(integer_t), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_xij   , xij   , n * nrhs * sizeof(complex_t), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_bij   , bij   , n * nrhs * sizeof(complex_t), cudaMemcpyHostToDevice ) );


	/* The handle must be initialized prior to calling any other library function */
	checkCudaErrors( cusolverSpCreate(&cusolverHandle));

	/* Create matrix descriptor */
	checkCudaErrors( cusparseSetMatType      ( MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL ) );
	checkCudaErrors( cusparseSetMatIndexBase ( MatDescr, CUSPARSE_INDEX_BASE_ZERO     ) );

	/* Verify if A is symmetric or not */
	checkCudaErrors( cusolverSpXcsrissymHost( cusolverHandle, 
		n, 
		nnz, 
		MatDescr, 
		rowptr, 
		rowptr +1, 
		colind, 
		&issym ));

	if ( issym ) fprintf(stderr, "Input matrix is symmetric.\n");

	/* Solve the linear system on the GPU using the high-level API    */
	/* We use QR factorization, LU factorization is not supported yet */
	checkCudaErrors( cusolverSpScsrlsvqr ( cusolverHandle,
	                n,
	                nnz,
	                MatDescr,
	                d_aij,
	                d_rowptr,
	                d_colind,
	                d_bij,
	                epsilon,
	                reorder,
	                d_xij,
	                &singular));

	/* transfer the solution back to the host memory */
	checkCudaErrors( cudaMemcpy( xij, d_xij, n * nrhs * sizeof(complex_t), cudaMemcpyDeviceToHost ));

	fprintf(stderr, "\nSolution of the linear system\n");
	for(integer_t i=0; i < n; i++ )
		fprintf(stderr, "x[%d] = %.6f\n", i, xij[i]);


	/* NOW, WE USE THE LOW-LEVEL API */
	csrqrInfo_t csrqrInfo = NULL;
	checkCudaErrors( cusolverSpCreateCsrqrInfo ( &csrqrInfo ));

	/* analyses sparsity pattern of H and Q matrices */
	checkCudaErrors( cusolverSpXcsrqrAnalysis ( cusolverHandle,
                           n,
                           n,
                           nnz,
                           MatDescr,
                           d_rowptr,
                           d_colind,
                           csrqrInfo ));

	/* After the analysis, the size of working space to perform QR factorization can be retrieved */
	size_t internalDataInBytes = 0;
	size_t workspaceInBytes    = 0;

	checkCudaErrors( cusolverSpScsrqrBufferInfo ( cusolverHandle,
                           n,
                           n,
                           nnz,
                           MatDescr,
                           d_aij,
                           d_rowptr,
                           d_colind,
                           csrqrInfo,
                           &internalDataInBytes,
                           &workspaceInBytes));

	fprintf(stderr, "\nInternal data in bytes : %lu", internalDataInBytes);
	fprintf(stderr, "\nWorkspace in bytes     : %lu", workspaceInBytes   );

	/* allocate space for the later factorization */
	void *d_work;
	checkCudaErrors( cudaMalloc((void**) &d_work, workspaceInBytes ));

	/* This function shifts diagonal of A by parameter mu such that we can factorize */
	/* For linear solver, the user just sets mu to zero.                             */ 
	/* For eigenvalue solver, mu can be a value of shift in inverse-power method.    */
	checkCudaErrors( cusolverSpScsrqrSetup ( cusolverHandle,
                           n,
                           n,
                           nnz,
                           MatDescr,
                           d_aij,
                           d_rowptr,
                           d_colind,
                           0,
                           csrqrInfo ));

	/* Perform numerical factorization */
	/* If either x or b is nil, only factorization is done. The user needs cusolverSpXcsrqrSolve    */
	/* to find the least-square solution.                                                           */
	/* If both x and b are not nil, QR factorization and solve are combined together. b is over-    */
	/* written by c and x is the solution of least-square.                                          */
	/* pBuffer: buffer allocated by the user, the size is returned by cusolverSpXcsrqrBufferInfo(). */
	checkCudaErrors( cusolverSpScsrqrFactor ( cusolverHandle,
                           n,
                           n,
                           nnz,
                           d_bij,
                           d_xij,
                           csrqrInfo,
                           d_work ));

//	/* Solve the system, only if it was only factorized */
//	checkCudaErrors( cusolverSpScsrqrSolve ( cusolverHandle,
//                           m,
//                           n,
//                           d_bij,
//                           d_xij,
//                           csrqrInfo,
//                           d_work ));

	/* transfer the solution back to the host memory */
	checkCudaErrors( cudaMemcpy( xij, d_xij, n * nrhs * sizeof(complex_t), cudaMemcpyDeviceToHost ));

	fprintf(stderr, "\nSolution of the linear system\n");
	for(integer_t i=0; i < n; i++ )
		fprintf(stderr, "x[%d] = %.6f\n", i, xij[i]);


	checkCudaErrors( cusolverSpDestroyCsrqrInfo ( csrqrInfo ));
	if ( d_work ) { cudaFree( d_work ); }





	/* clean up and */
	if ( cusolverHandle ) { checkCudaErrors( cusolverSpDestroy(cusolverHandle)); }
	if ( cusparseHandle ) { checkCudaErrors( cusparseDestroy  (cusparseHandle)); }
	if ( MatDescr       ) { checkCudaErrors( cusparseDestroyMatDescr(MatDescr)); }

	if ( d_aij    ) { checkCudaErrors( cudaFree( d_aij    )); }
	if ( d_colind ) { checkCudaErrors( cudaFree( d_colind )); }
	if ( d_rowptr ) { checkCudaErrors( cudaFree( d_rowptr )); }
	if ( d_xij    ) { checkCudaErrors( cudaFree( d_xij    )); }
	if ( d_bij    ) { checkCudaErrors( cudaFree( d_bij    )); }

	cudaDeviceReset();


	fprintf(stderr, "Testing cuSolver High Level Interface\n");

	return 0;
}