#include "spike_cuda.h"

unsigned int cnt_devMalloc = 0;
unsigned int cnt_devFree   = 0;

/* ------------------------------------------------------------------------- */
/* .. These macros are used to build the name of the API function according  */
/* to the numerical precision at hand (i.e. single, double, etc.)            */
/* ------------------------------------------------------------------------- */
#define CAT_I(a,b,c) a##b##c
#define CAT(a,b,c)   CAT_I(a,b,c)
#define CUDA_KERNEL(lib,prec,call) CAT(lib,prec,call)

static inline void checkCudaErrors(cudaError_t error, const char* function, const int line )
{
	if ( error != cudaSuccess ){
		char* what;

	    switch (error)
	    {
	        case cudaSuccess:
	            what = "cudaSuccess";
	            break;
	        case cudaErrorMissingConfiguration:
	            what = "cudaErrorMissingConfiguration";
	            break;
	        case cudaErrorMemoryAllocation:
	            what = "cudaErrorMemoryAllocation";
	            break;
	        case cudaErrorInitializationError:
	            what = "cudaErrorInitializationError";
	            break;
	        case cudaErrorLaunchFailure:
	            what = "cudaErrorLaunchFailure";
	            break;
	        case cudaErrorPriorLaunchFailure:
	            what = "cudaErrorPriorLaunchFailure";
	            break;
	        case cudaErrorLaunchTimeout:
	            what = "cudaErrorLaunchTimeout";
	            break;
	        case cudaErrorLaunchOutOfResources:
	            what = "cudaErrorLaunchOutOfResources";
	            break;
	        case cudaErrorInvalidDeviceFunction:
	            what = "cudaErrorInvalidDeviceFunction";
	            break;
	        case cudaErrorInvalidConfiguration:
	            what = "cudaErrorInvalidConfiguration";
	            break;
	        case cudaErrorInvalidDevice:
	            what = "cudaErrorInvalidDevice";
	            break;
	        case cudaErrorInvalidValue:
	            what = "cudaErrorInvalidValue";
	            break;
	        case cudaErrorInvalidPitchValue:
	            what = "cudaErrorInvalidPitchValue";
	            break;
	        case cudaErrorInvalidSymbol:
	            what = "cudaErrorInvalidSymbol";
	            break;
	        case cudaErrorMapBufferObjectFailed:
	            what = "cudaErrorMapBufferObjectFailed";
	            break;
	        case cudaErrorUnmapBufferObjectFailed:
	            what = "cudaErrorUnmapBufferObjectFailed";
	            break;
	        case cudaErrorInvalidHostPointer:
	            what = "cudaErrorInvalidHostPointer";
	            break;
	        case cudaErrorInvalidDevicePointer:
	            what = "cudaErrorInvalidDevicePointer";
	            break;
	        case cudaErrorInvalidTexture:
	            what = "cudaErrorInvalidTexture";
	            break;
	        case cudaErrorInvalidTextureBinding:
	            what = "cudaErrorInvalidTextureBinding";
	            break;
	        case cudaErrorInvalidChannelDescriptor:
	            what = "cudaErrorInvalidChannelDescriptor";
	            break;
	        case cudaErrorInvalidMemcpyDirection:
	            what = "cudaErrorInvalidMemcpyDirection";
	            break;
	        case cudaErrorAddressOfConstant:
	            what = "cudaErrorAddressOfConstant";
	            break;
	        case cudaErrorTextureFetchFailed:
	            what = "cudaErrorTextureFetchFailed";
	            break;
	        case cudaErrorTextureNotBound:
	            what = "cudaErrorTextureNotBound";
	            break;
	        case cudaErrorSynchronizationError:
	            what = "cudaErrorSynchronizationError";
	            break;
	        case cudaErrorInvalidFilterSetting:
	            what = "cudaErrorInvalidFilterSetting";
	            break;
	        case cudaErrorInvalidNormSetting:
	            what = "cudaErrorInvalidNormSetting";
	            break;
	        case cudaErrorMixedDeviceExecution:
	            what = "cudaErrorMixedDeviceExecution";
	            break;
	        case cudaErrorCudartUnloading:
	            what = "cudaErrorCudartUnloading";
	            break;
	        case cudaErrorUnknown:
	            what = "cudaErrorUnknown";
	            break;
	        case cudaErrorNotYetImplemented:
	            what = "cudaErrorNotYetImplemented";
	            break;
	        case cudaErrorMemoryValueTooLarge:
	            what = "cudaErrorMemoryValueTooLarge";
	            break;
	        case cudaErrorInvalidResourceHandle:
	            what = "cudaErrorInvalidResourceHandle";
	            break;
	        case cudaErrorNotReady:
	            what = "cudaErrorNotReady";
	            break;
	        case cudaErrorInsufficientDriver:
	            what = "cudaErrorInsufficientDriver";
	            break;
	        case cudaErrorSetOnActiveProcess:
	            what = "cudaErrorSetOnActiveProcess";
	            break;
	        case cudaErrorInvalidSurface:
	            what = "cudaErrorInvalidSurface";
	            break;
	        case cudaErrorNoDevice:
	            what = "cudaErrorNoDevice";
	            break;
	        case cudaErrorECCUncorrectable:
	            what = "cudaErrorECCUncorrectable";
	            break;
	        case cudaErrorSharedObjectSymbolNotFound:
	            what = "cudaErrorSharedObjectSymbolNotFound";
	            break;
	        case cudaErrorSharedObjectInitFailed:
	            what = "cudaErrorSharedObjectInitFailed";
	            break;
	        case cudaErrorUnsupportedLimit:
	            what = "cudaErrorUnsupportedLimit";
	            break;
	        case cudaErrorDuplicateVariableName:
	            what = "cudaErrorDuplicateVariableName";
	            break;
	        case cudaErrorDuplicateTextureName:
	            what = "cudaErrorDuplicateTextureName";
	            break;
	        case cudaErrorDuplicateSurfaceName:
	            what = "cudaErrorDuplicateSurfaceName";
	            break;
	        case cudaErrorDevicesUnavailable:
	            what = "cudaErrorDevicesUnavailable";
	            break;
	        case cudaErrorInvalidKernelImage:
	            what = "cudaErrorInvalidKernelImage";
	            break;
	        case cudaErrorNoKernelImageForDevice:
	            what = "cudaErrorNoKernelImageForDevice";
	            break;
	        case cudaErrorIncompatibleDriverContext:
	            what = "cudaErrorIncompatibleDriverContext";
	            break;
	        case cudaErrorPeerAccessAlreadyEnabled:
	            what = "cudaErrorPeerAccessAlreadyEnabled";
	            break;
	        case cudaErrorPeerAccessNotEnabled:
	            what = "cudaErrorPeerAccessNotEnabled";
	            break;
	        case cudaErrorDeviceAlreadyInUse:
	            what = "cudaErrorDeviceAlreadyInUse";
	            break;
	        case cudaErrorProfilerDisabled:
	            what = "cudaErrorProfilerDisabled";
	            break;
	        case cudaErrorProfilerNotInitialized:
	            what = "cudaErrorProfilerNotInitialized";
	            break;
	        case cudaErrorProfilerAlreadyStarted:
	            what = "cudaErrorProfilerAlreadyStarted";
	            break;
	        case cudaErrorProfilerAlreadyStopped:
	            what = "cudaErrorProfilerAlreadyStopped";
	            break;

	        /* Since CUDA 4.0*/
	        case cudaErrorAssert:
	            what = "cudaErrorAssert";
	            break;
	        case cudaErrorTooManyPeers:
	            what = "cudaErrorTooManyPeers";
	            break;
	        case cudaErrorHostMemoryAlreadyRegistered:
	            what = "cudaErrorHostMemoryAlreadyRegistered";
	            break;
	        case cudaErrorHostMemoryNotRegistered:
	            what = "cudaErrorHostMemoryNotRegistered";
	            break;

	        /* Since CUDA 5.0 */
	        case cudaErrorOperatingSystem:
	            what = "cudaErrorOperatingSystem";
	            break;
	        case cudaErrorPeerAccessUnsupported:
	            what = "cudaErrorPeerAccessUnsupported";
	            break;
	        case cudaErrorLaunchMaxDepthExceeded:
	            what = "cudaErrorLaunchMaxDepthExceeded";
	            break;
	        case cudaErrorLaunchFileScopedTex:
	            what = "cudaErrorLaunchFileScopedTex";
	            break;
	        case cudaErrorLaunchFileScopedSurf:
	            what = "cudaErrorLaunchFileScopedSurf";
	            break;
	        case cudaErrorSyncDepthExceeded:
	            what = "cudaErrorSyncDepthExceeded";
	            break;
	        case cudaErrorLaunchPendingCountExceeded:
	            what = "cudaErrorLaunchPendingCountExceeded";
	            break;
	        case cudaErrorNotPermitted:
	            what = "cudaErrorNotPermitted";
	            break;
	        case cudaErrorNotSupported:
	            what = "cudaErrorNotSupported";
	            break;

	        /* Since CUDA 6.0 */
	        case cudaErrorHardwareStackError:
	            what = "cudaErrorHardwareStackError";
				break;
	        case cudaErrorIllegalInstruction:
	            what = "cudaErrorIllegalInstruction";
				break;
	        case cudaErrorMisalignedAddress:
	            what = "cudaErrorMisalignedAddress";
				break;
	        case cudaErrorInvalidAddressSpace:
	            what = "cudaErrorInvalidAddressSpace";
				break;
	        case cudaErrorInvalidPc:
	            what = "cudaErrorInvalidPc";
				break;
	        case cudaErrorIllegalAddress:
	            what = "cudaErrorIllegalAddress";
				break;
	        /* Since CUDA 6.5*/
	        case cudaErrorInvalidPtx:
	            what = "cudaErrorInvalidPtx";
				break;
	        case cudaErrorInvalidGraphicsContext:
	            what = "cudaErrorInvalidGraphicsContext";
				break;
	        case cudaErrorStartupFailure:
	            what = "cudaErrorStartupFailure";
				break;
	        case cudaErrorApiFailureBase:
	            what = "cudaErrorApiFailureBase";
	        	break;
	        default:
	        	what = "undefined error";
	        	break;
	    }
	    fprintf(stderr, "\n%s:%d cusolver status %d (%s)\n", function, line, (int) error, what );
		abort();			
	}
}

static inline void cusolverSpCheck( cusolverStatus_t status, const char* function, const int line )
{
	if ( status != CUSOLVER_STATUS_SUCCESS ) {
		char *what;
		
		switch( status ){
			case CUSOLVER_STATUS_NOT_INITIALIZED:
				what = "status not initialized";
				break;
			case CUSOLVER_STATUS_ALLOC_FAILED:
				what = "status alloc failed";
				break;
			case CUSOLVER_STATUS_INVALID_VALUE:
				what = "status invalid value";
				break;
			case CUSOLVER_STATUS_ARCH_MISMATCH:
				what = "status arch mismatch";
				break;
			case CUSOLVER_STATUS_MAPPING_ERROR:
				what = "status mapping error";
				break;
			case CUSOLVER_STATUS_EXECUTION_FAILED:
				what = "status execution failed";
				break;
			case CUSOLVER_STATUS_INTERNAL_ERROR:
				what = "status internal error";
				break;
			case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
				what = "status matrix type not supported";
				break;
			case CUSOLVER_STATUS_NOT_SUPPORTED:
				what = "status not supported";
				break;
			case CUSOLVER_STATUS_ZERO_PIVOT:
				what = "status zero pivot";
				break;
			case CUSOLVER_STATUS_INVALID_LICENSE:
				what = "status invalid license";
				break;
			default:
				what = "status undefined error";
				break;
		}

		fprintf(stderr, "\n%s:%d cusolver status %d (%s)\n", function, line, (int) status, what );
		abort();
	}
};

static inline int spike_devMalloc( void *devPtr, const size_t nmemb, const size_t size )
{
	cnt_devMalloc += 1;
	
	return( cudaMalloc((void**) &devPtr, nmemb * size ));
};

static inline int spike_devNullify( void* devPtr )
{
	cnt_devFree += 1;

	if( devPtr) return ( cudaFree( devPtr ));

	return 0;
};


DirectSolverHander_t *directSolver_CreateHandler(void)
{

	// TODO check CUDA version >= 7.5

	DirectSolverHander_t *handler = (DirectSolverHander_t*) spike_malloc( ALIGN_INT, 1, sizeof(DirectSolverHander_t));

	return (handler);
};

Error_t directSolver_Configure( DirectSolverHander_t *handler, const integer_t max_nrhs )
{
	/* -------------------------------------------------------------------- */
	/* .. Query used and free memory on the device. */
	/* -------------------------------------------------------------------- */	
	checkCudaErrors( cudaMemGetInfo( &handler->deviceFreeMemory, &handler->deviceTotalMemory ), "cudaMemGetInfo", __LINE__  );

	fprintf(stderr, "\n\t Device memory at solver configuration stage:");
	fprintf(stderr, "\n\t\t Device free memory : %5.2lf GB", bytesToGb(handler->deviceFreeMemory) );
	fprintf(stderr, "\n\t\t Device free memory : %5.2lf GB", bytesToGb(handler->deviceFreeMemory) );

	/* -------------------------------------------------------------------- */
	/* .. Initilialize handlers.                                            */
	/* -------------------------------------------------------------------- */
	cusolverSpCheck( cusolverSpCreate          ( &handler->cusolverHandle ), "cusolverSpCreate"         , __LINE__ );
	cusolverSpCheck( cusolverSpCreateCsrqrInfo ( &handler->csrqrInfo      ), "cusolverSpCreateCsrqrInfo", __LINE__ );

	/* -------------------------------------------------------------------- */
	/* .. Set current CUDA stream                                           */
	/* -------------------------------------------------------------------- */
	// handler->stream = (cudaStream_t*) spike_malloc( ALIGN_INT, 1, sizeof(cudaStream_t));
	checkCudaErrors( cudaStreamCreate( &handler->stream[0] ), "cudaStreamCreate", __LINE__ );

	/* -------------------------------------------------------------------- */
	/* .. No device memory is allocated for RHS blocks                      */ 
	/* .. Set the number of RHS to a default value                          */
	/* -------------------------------------------------------------------- */
	handler->deviceMemForRHSisAllocated   =    False;
	handler->max_nrhs                     = max_nrhs;    /* max number of rhs to solve for any linear sub-system */
	handler->deviceRHSBlockDist           =       300;          /* optimal transfer to solve ratio */

	/* -------------------------------------------------------------------- */
	/* .. Associate CUDA stream to cusolver library                         */
	/* -------------------------------------------------------------------- */
	cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );

	/* -------------------------------------------------------------------- */
	/* .. Setup matrix information.                                         */
	/* -------------------------------------------------------------------- */
	checkCudaErrors( cusparseCreateMatDescr  ( &handler->MatDescr), "cusparseCreateMatDescr", __LINE__ );
	checkCudaErrors( cusparseSetMatType      ( handler->MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL), "cusparseSetMatType", __LINE__ );
	checkCudaErrors( cusparseSetMatIndexBase ( handler->MatDescr, CUSPARSE_INDEX_BASE_ZERO), "cusparseSetMatIndexBase", __LINE__ );

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
	/* local variables */
	integer_t i; 

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
	checkCudaErrors( cudaMalloc((void**) &handler->d_aij   , handler->nnz    * sizeof(complex_t)), "cudaMalloc", __LINE__);
	checkCudaErrors( cudaMalloc((void**) &handler->d_colind, handler->nnz    * sizeof(integer_t)), "cudaMalloc", __LINE__);
	checkCudaErrors( cudaMalloc((void**) &handler->d_rowptr, (handler->n +1) * sizeof(integer_t)), "cudaMalloc", __LINE__);

	/* transfer the arrays to the device memory */
	checkCudaErrors( cudaMemcpyAsync( handler->d_aij   , handler->h_aij   ,  handler->nnz    * sizeof(complex_t), cudaMemcpyHostToDevice, handler->stream[0] ), "cudaMemcpyAsync", __LINE__ );
	checkCudaErrors( cudaMemcpyAsync( handler->d_colind, handler->h_colind,  handler->nnz    * sizeof(integer_t), cudaMemcpyHostToDevice, handler->stream[0] ), "cudaMemcpyAsync", __LINE__ );
	checkCudaErrors( cudaMemcpyAsync( handler->d_rowptr, handler->h_rowptr, (handler->n + 1) * sizeof(integer_t), cudaMemcpyHostToDevice, handler->stream[0] ), "cudaMemcpyAsync", __LINE__ );


	/* -------------------------------------------------------------------- */
	/* .. Verify if A is symmetric.                                         */
	/* -------------------------------------------------------------------- */
//    checkCudaErrors(cusolverSpXcsrissymHost( handler->cusolverHandle,
//    						handler->n,
//    						handler->nnz,
//    						handler->MatDescr,
//    						handler->h_rowptr,
//    						handler->h_rowptr +1,
//    						handler->h_colind,
//    						&handler->issym));
	
	/* -------------------------------------------------------------------- */
	/* .. Reorder to reduce fill-in.                                        */
	/*                                                                      */
	/* The low-level API does not reorder under the hood, so it is needed   */
	/* to do so explicitely.                                                */
	/* -------------------------------------------------------------------- */   

    // TODO.

	/* analyses sparsity pattern of H and Q matrices */
	cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
	cusolverSpCheck( cusolverSpXcsrqrAnalysis ( handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           handler->MatDescr,
                           handler->d_rowptr,
                           handler->d_colind,
                           handler->csrqrInfo ), "cusolverSpXcsrqrAnalysis", __LINE__);

	/* After the analysis, the size of working space to perform QR factorization can be retrieved */
	cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
	cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrBufferInfo ) ( 
						   handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           handler->MatDescr,
                           (const complex_t*) handler->d_aij,
                           handler->d_rowptr,
                           handler->d_colind,
                           handler->csrqrInfo,
                           &handler->internalDataInBytes,
                           &handler->workspaceInBytes), "csrqrBufferInfo", __LINE__);


	/* -------------------------------------------------------------------- */
	/* .. Check if there is enough space for factorization on the GPU.      */
	/* If so, allocate the work buffer there                                */
	/* -------------------------------------------------------------------- */	
	checkCudaErrors( cudaMemGetInfo( &handler->deviceFreeMemory, &handler->deviceTotalMemory ), "cudaMemGetInfo", __LINE__  );

	if( handler->deviceFreeMemory < handler->workspaceInBytes ){
		fprintf(stderr, "\n ERROR: there is not enough space on the GPU for factorization.");
		abort();
	} else {
		fprintf(stderr, "\n Device memory after factorization will be %.f GB", bytesToGb( handler->deviceFreeMemory - handler->workspaceInBytes ) );
		checkCudaErrors( cudaMalloc((void**) &handler->d_work, handler->workspaceInBytes), "cudaMalloc", __LINE__ );
	}
	
	/* This function shifts diagonal of A by parameter mu such that we can factorize */
	/* For linear solver, the user just sets mu to zero.                             */ 
	/* For eigenvalue solver, mu can be a value of shift in inverse-power method.    */
	cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
	cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSetup ) ( 
						   handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           handler->MatDescr,
                           (const complex_t*) handler->d_aij,
                           handler->d_rowptr,
                           handler->d_colind,
                           0,
                           handler->csrqrInfo ), "csrqrSetup", __LINE__);

	/* Perform numerical factorization */
	/* If either x or b is nil, only factorization is done. The user needs cusolverSpXcsrqrSolve    */
	/* to find the least-square solution.                                                           */
	/* If both x and b are not nil, QR factorization and solve are combined together. b is over-    */
	/* written by c and x is the solution of least-square.                                          */
	/* pBuffer: buffer allocated by the user, the size is returned by cusolverSpXcsrqrBufferInfo(). */
	cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
	cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrFactor ) ( 
						   handler->cusolverHandle,
                           handler->n,
                           handler->n,
                           handler->nnz,
                           NULL, /* handler->d_bij, */
                           NULL, /* handler->d_xij, */
                           handler->csrqrInfo,
                           handler->d_work ), "csrqrFactor", __LINE__);

	/* Deallocate matrix from device */
	checkCudaErrors( spike_devNullify( handler->d_colind), "cudaFree", __LINE__ );
	checkCudaErrors( spike_devNullify( handler->d_rowptr), "cudaFree", __LINE__ );
	checkCudaErrors( spike_devNullify( handler->d_aij   ), "cudaFree", __LINE__ );

	/* -------------------------------------------------------------------- */
	/* .. Query used and free memory on the device. */
	/* -------------------------------------------------------------------- */	
	checkCudaErrors( cudaMemGetInfo( &handler->deviceFreeMemory, &handler->deviceTotalMemory ), "cudaMemGetInfo", __LINE__  );

	/* compute the number of rhs that fits into the device memory */
	handler->nrhs_fiffing = floor(  ((double) handler->deviceFreeMemory) / ((double) 2.f * handler->n * sizeof(complex_t)));
	fprintf(stderr, "\nFree mem %5.2f GB. There is room for %d rhs on the device (%lf GB)", 
		bytesToGb( handler->deviceFreeMemory), handler->nrhs_fiffing, bytesToGb( 2. * handler->n * handler->nrhs * sizeof(complex_t)));

	/* Correct the blocking distance according to the number of RHS fitting on the device */
	if ( handler->nrhs_fiffing < handler->deviceRHSBlockDist ){
		handler->deviceRHSBlockDist = handler->nrhs_fiffing;
		fprintf(stderr, "\n GPU col blocking has been corrected due to memory space issues.");
	}

	/* compute the number of streams to be used */
	if ( handler->max_nrhs % handler->deviceRHSBlockDist == 0 ){
		handler->nstreams = handler->max_nrhs / handler->deviceRHSBlockDist;
	}
	else {
		handler->nstreams = ceil( ((double) handler->max_nrhs) / ((double) handler->deviceRHSBlockDist));
	}

	fprintf(stderr, "\n%s: %d streams will be used for solving a maximum number of %d rhs using a blocking distance of %d rhs\n", __FUNCTION__, handler->nstreams, handler->max_nrhs, handler->deviceRHSBlockDist );


	if ( handler->deviceMemForRHSisAllocated == False ) {
		/* allocate space for one x column and for one b column  */
		checkCudaErrors( cudaMalloc((void**) &handler->d_xij, handler->n * handler->deviceRHSBlockDist * handler->nstreams * sizeof(complex_t)), "cudaMalloc", __LINE__);
		checkCudaErrors( cudaMalloc((void**) &handler->d_bij, handler->n * handler->deviceRHSBlockDist * handler->nstreams * sizeof(complex_t)), "cudaMalloc", __LINE__);

		/* set the flag to true for the next time */
		handler->deviceMemForRHSisAllocated = True;
	}

	/* Create additional streams if needed */
	if ( handler->nstreams > 1 ){
		// handler->stream = (cudaStream_t*) realloc( handler->stream, handler->nstreams * sizeof(cudaStream_t));

		for(i=1; i < handler->nstreams; i++ )
			checkCudaErrors( cudaStreamCreate( &handler->stream[i] ), "cudaStreamCreate", __LINE__ );

		fprintf(stderr, "\n %d CUDA streams were created\n", handler->nstreams );
	}

	/* resume and return */
	return (SPIKE_SUCCESS);
};

Error_t directSolver_SolveForRHS ( DirectSolverHander_t* handler,
                                const integer_t nrhs,
                                complex_t *__restrict__ xij,
                                complex_t *__restrict__ bij)
{
	directSolver_SolveForRHS_async_1 ( handler, nrhs, xij, bij);
	// directSolver_SolveForRHS_sync    ( handler, nrhs, xij, bij);

	return (SPIKE_SUCCESS);
}

Error_t directSolver_SolveForRHS_async_1 ( DirectSolverHander_t* handler,
                            const integer_t nrhs,
                            complex_t *__restrict__ xij,
                            complex_t *__restrict__ bij)
{
		/* local variables */
	int rhsCol = 0, rhs = 0, currentStream = 0;

	/* update the value of rhs columns */
	handler->nrhs = nrhs;

	/* update statistics, keep track of RHS */
	handler->rhs_block_count  += 1;
	handler->rhs_column_count += nrhs;

	if ( handler->nrhs <= handler->deviceRHSBlockDist ){
		fprintf(stderr, "\n First case, nrhs < blocking distance");
		/* Transfer the arrays to the device memory */
		checkCudaErrors( cudaMemcpyAsync( handler->d_bij, bij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyHostToDevice, handler->stream[0] ), "cudaMemcpyAsync", __LINE__);
		
		for(rhs=0; rhs < nrhs; rhs++ ) {
			/* Forward and backward substitution */
			cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
			cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
								   handler->cusolverHandle,
		                           handler->n,
		                           handler->n,
		                           &handler->d_bij[ rhs * handler->n ],
		                           &handler->d_xij[ rhs * handler->n ],
		                           handler->csrqrInfo,
		                           handler->d_work ), "csrqrSolve", __LINE__);
		}
		
		/* transfer the solution back to the host */
		checkCudaErrors( cudaMemcpyAsync( xij, handler->d_xij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyDeviceToHost, handler->stream[0] ), "cudaMemcpyAsync", __LINE__);

		fprintf(stderr, "\nFin del caso");
	}
	else{
		for( rhsCol = 0; (rhsCol + handler->deviceRHSBlockDist) <= handler->nrhs; rhsCol += handler->deviceRHSBlockDist ){

			/* set current stream */
			currentStream = ( (currentStream +1) == handler->nstreams ) ? 0 : currentStream + 1;

			fprintf(stderr, "\n Current stream %d", currentStream );

			/* Transfer the arrays to the device memory */
			checkCudaErrors( cudaMemcpyAsync( handler->d_bij, 
				&bij[rhsCol * handler->n], 
				handler->n  * handler->deviceRHSBlockDist * sizeof(complex_t),
				cudaMemcpyHostToDevice,
				handler->stream[currentStream]),
				"cudaMemcpy", __LINE__);
		
			for(rhs=0; rhs < handler->deviceRHSBlockDist; rhs++ ) {
				/* Forward and backward substitution */
				cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[currentStream] ), "cusolverSpSetStream", __LINE__ );
				cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
									   handler->cusolverHandle,
			                           handler->n,
			                           handler->n,
			                           &handler->d_bij[ rhs * handler->n ],
			                           &handler->d_xij[ rhs * handler->n ],
			                           handler->csrqrInfo,
			                           handler->d_work ), "csrqrSolve", __LINE__);
			}
		
			/* transfer the solution back to the host */
			checkCudaErrors( cudaMemcpyAsync( &xij[rhsCol * handler->n ],
				handler->d_xij,
				handler->n * handler->deviceRHSBlockDist * sizeof(complex_t),
				cudaMemcpyDeviceToHost,
				handler->stream[currentStream]),
				"cudaMemcpy", __LINE__);
		}



		if ( rhsCol < handler->nrhs ){
			/* set current stream */
			currentStream = ( (currentStream +1) == handler->nstreams ) ? 0 : currentStream + 1;
			fprintf(stderr, "\n Current stream %d", currentStream );

			/* number of colums to process */
			integer_t rhsToProcess = handler->nrhs - rhsCol;

			/* Transfer the arrays to the device memory */
			checkCudaErrors( cudaMemcpy( handler->d_bij, &bij[rhsCol * handler->n], handler->n * rhsToProcess * sizeof(complex_t), cudaMemcpyHostToDevice ), "cudaMemcpy", __LINE__);
		
			for(rhs=0; rhs < rhsToProcess; rhs++ ) {
				/* Forward and backward substitution */
				cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
				cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
									   handler->cusolverHandle,
			                           handler->n,
			                           handler->n,
			                           &handler->d_bij[ rhs * handler->n ],
			                           &handler->d_xij[ rhs * handler->n ],
			                           handler->csrqrInfo,
			                           handler->d_work ), "csrqrSolve", __LINE__);
			}
		
			/* transfer the solution back to the host */
			checkCudaErrors( cudaMemcpy( &xij[rhsCol * handler->n ], handler->d_xij, handler->n * rhsToProcess * sizeof(complex_t), cudaMemcpyDeviceToHost ), "cudaMemcpy", __LINE__);
		}
	}

	return (SPIKE_SUCCESS);
};

Error_t directSolver_SolveForRHS_async_0 ( DirectSolverHander_t* handler,
                            const integer_t nrhs,
                            complex_t *__restrict__ xij,
                            complex_t *__restrict__ bij)
{
	/* local variables */
	int rhsCol = 0, rhs = 0;

	/* update the value of rhs columns */
	handler->nrhs = nrhs;

	/* update statistics, keep track of RHS */
	handler->rhs_block_count  += 1;
	handler->rhs_column_count += nrhs;

	if ( handler->nrhs <= handler->deviceRHSBlockDist){
		/* Transfer the arrays to the device memory */
		checkCudaErrors( cudaMemcpy( handler->d_bij, bij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyHostToDevice ), "cudaMemcpy", __LINE__);
		
		for(rhs=0; rhs < nrhs; rhs++ ) {
			/* Forward and backward substitution */
			cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
								   handler->cusolverHandle,
		                           handler->n,
		                           handler->n,
		                           &handler->d_bij[ rhs * handler->n ],
		                           &handler->d_xij[ rhs * handler->n ],
		                           handler->csrqrInfo,
		                           handler->d_work ), "csrqrSolve", __LINE__);
		}
		
		/* transfer the solution back to the host */
		checkCudaErrors( cudaMemcpy( xij, handler->d_xij, handler->n * handler->nrhs * sizeof(complex_t), cudaMemcpyDeviceToHost ), "cudaMemcpy", __LINE__);
	}
	else{
		for( rhsCol = 0; (rhsCol + handler->deviceRHSBlockDist) <= handler->nrhs; rhsCol += handler->deviceRHSBlockDist ){
			/* Transfer the arrays to the device memory */
			checkCudaErrors( cudaMemcpy( handler->d_bij, &bij[rhsCol * handler->n], handler->n * handler->deviceRHSBlockDist * sizeof(complex_t), cudaMemcpyHostToDevice ), "cudaMemcpy", __LINE__);
		
			for(rhs=0; rhs < handler->deviceRHSBlockDist; rhs++ ) {
				/* Forward and backward substitution */
				cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
									   handler->cusolverHandle,
			                           handler->n,
			                           handler->n,
			                           &handler->d_bij[ rhs * handler->n ],
			                           &handler->d_xij[ rhs * handler->n ],
			                           handler->csrqrInfo,
			                           handler->d_work ), "csrqrSolve", __LINE__);
			}
		
			/* transfer the solution back to the host */
			checkCudaErrors( cudaMemcpy( &xij[rhsCol * handler->n ], handler->d_xij, handler->n * handler->deviceRHSBlockDist * sizeof(complex_t), cudaMemcpyDeviceToHost ), "cudaMemcpy", __LINE__);
		}

		if ( rhsCol < handler->nrhs ){
			/* number of colums to process */
			integer_t rhsToProcess = handler->nrhs - rhsCol;

			/* Transfer the arrays to the device memory */
			checkCudaErrors( cudaMemcpy( handler->d_bij, &bij[rhsCol * handler->n], handler->n * rhsToProcess * sizeof(complex_t), cudaMemcpyHostToDevice ), "cudaMemcpy", __LINE__);
		
			for(rhs=0; rhs < rhsToProcess; rhs++ ) {
				/* Forward and backward substitution */
				cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
									   handler->cusolverHandle,
			                           handler->n,
			                           handler->n,
			                           &handler->d_bij[ rhs * handler->n ],
			                           &handler->d_xij[ rhs * handler->n ],
			                           handler->csrqrInfo,
			                           handler->d_work ), "csrqrSolve", __LINE__);
			}
		
			/* transfer the solution back to the host */
			checkCudaErrors( cudaMemcpy( &xij[rhsCol * handler->n ], handler->d_xij, handler->n * rhsToProcess * sizeof(complex_t), cudaMemcpyDeviceToHost ), "cudaMemcpy", __LINE__);
		}
	}

	return (SPIKE_SUCCESS);
};

Error_t directSolver_SolveForRHS_sync ( DirectSolverHander_t* handler,
                            const integer_t nrhs,
                            complex_t *__restrict__ xij,
                            complex_t *__restrict__ bij)
{
	/* local variables */
	int rhsCol = 0;

	/* update the value of rhs columns */
	handler->nrhs = nrhs;

	/* update statistics, keep track of RHS */
	handler->rhs_block_count  += 1;
	handler->rhs_column_count += nrhs;

	/* At the moment, the API is not able to handle multiple */
	/* rhs at a time, so we have to iterate                 */
	for(rhsCol=0; rhsCol < nrhs; rhsCol++ ) {
		/* Transfer the arrays to the device memory */
		//checkCudaErrors( cudaMemcpyAsync( handler->d_xij, &xij[rhsCol * handler->n], handler->n * sizeof(complex_t), cudaMemcpyHostToDevice, handler->stream ), "cudaMemcpyAsync", __LINE__);
		checkCudaErrors( cudaMemcpy( handler->d_bij, &bij[rhsCol * handler->n], handler->n * sizeof(complex_t), cudaMemcpyHostToDevice ), "cudaMemcpy", __LINE__);

		/* Forward and backward substitution */
		cusolverSpCheck( cusolverSpSetStream( handler->cusolverHandle, handler->stream[0] ), "cusolverSpSetStream", __LINE__ );
		cusolverSpCheck( CUDA_KERNEL( cusolverSp,SPIKE_CUDA_PREC,csrqrSolve ) ( 
							   handler->cusolverHandle,
	                           handler->n,
	                           handler->n,
	                           handler->d_bij,
	                           handler->d_xij,
	                           handler->csrqrInfo,
	                           handler->d_work ), "csrqrSolve", __LINE__);

		/* transfer the solution back to the host */
		checkCudaErrors( cudaMemcpy( &xij[rhsCol * handler->n], handler->d_xij, handler->n * sizeof(complex_t), cudaMemcpyDeviceToHost ), "cudaMemcpy", __LINE__);
	}

	/* resume and return */
	return (SPIKE_SUCCESS);
};

Error_t directSolver_ShowStatistics( DirectSolverHander_t *handler )
{



	return (SPIKE_SUCCESS);
};

Error_t directSolver_Finalize( DirectSolverHander_t *handler )
{
	/* local variables */
	integer_t i;

	/* synchronize all streams */
	cudaDeviceSynchronize();
  	
fprintf(stderr, "\n%s:%d destroy cuda streams", __FUNCTION__, __LINE__ );

	/* synchronize and destroy streams */
	for(i=0; i < handler->nstreams; i++ ){
  		checkCudaErrors( cudaStreamSynchronize( handler->stream[i] ), "cudaStreamSynchronize", __LINE__ );
  		checkCudaErrors( cudaStreamDestroy    ( handler->stream[i] ), "cudaStreamDestroy", __LINE__ );

	}

fprintf(stderr, "\n%s:%d destroy cuda streams", __FUNCTION__, __LINE__ );


	/* deallocate device memory */
	checkCudaErrors( spike_devNullify( handler->d_work  ), "cudaFree", __LINE__ );
	checkCudaErrors( spike_devNullify( handler->d_xij   ), "cudaFree", __LINE__ );
	checkCudaErrors( spike_devNullify( handler->d_bij   ), "cudaFree", __LINE__ );

	/* destroy cusolverSP and cusparse handlers */
	checkCudaErrors( cusparseDestroyMatDescr( handler->MatDescr      ), "cusparseDestroyMatDescr"   , __LINE__ );
	cusolverSpCheck( cusolverSpDestroyCsrqrInfo ( handler->csrqrInfo ), "cusolverSpDestroyCsrqrInfo", __LINE__ );
	checkCudaErrors( cusolverSpDestroy      ( handler->cusolverHandle), "cusolverSpDestroy"         , __LINE__ );

	/* destroy CUDA stream array */
// spike_nullify(handler->stream);

	/* deallocate directSolver handler */
	spike_nullify(handler);

	fprintf(stderr, "\nSolver finalize");

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

Error_t cuda_standalone_symrcm (integer_t n,
    							integer_t nnz,
    							integer_t *__restrict__ rowptr,
    							integer_t *__restrict__ colind)
{
	/* local variables */
	size_t bufferSizeInBytes = 0;
	integer_t *p             = NULL;
	integer_t *pBuffer       = NULL;
	integer_t *map           = NULL;

	p   = (integer_t*) spike_malloc( ALIGN_INT, n  , sizeof(integer_t));
	map = (integer_t*) spike_malloc( ALIGN_INT, nnz, sizeof(integer_t));

	/* cuSolver handler */
	cusolverSpHandle_t cusolverHandle = NULL;

	/* cuSparse matrix descriptor */
	cusparseMatDescr_t MatDescr = NULL;

	/* The handle must be initialized prior to calling any other library function */
	checkCudaErrors( cusolverSpCreate(&cusolverHandle), "cusolverSpCreate", __LINE__ );

	/* Create matrix descriptor */
	checkCudaErrors( cusparseCreateMatDescr  ( &MatDescr), "cusparseCreateMatDescr", __LINE__ );
	checkCudaErrors( cusparseSetMatType      ( MatDescr, CUSPARSE_MATRIX_TYPE_GENERAL ), "cusparseSetMatType", __LINE__ );
	checkCudaErrors( cusparseSetMatIndexBase ( MatDescr, CUSPARSE_INDEX_BASE_ZERO ), "cusparseSetMatIndexBase", __LINE__ );

	/* use reverse cuthill mkee algorithm to obtain a bandwidth minization permutation */
	cusolverSpCheck( cusolverSpXcsrsymrcmHost(cusolverHandle,
		n, 
		nnz, 
		MatDescr, 
		rowptr, 
		colind, 
		p),
		"cusolverSpXcsrsymrcmHost", __LINE__ );

	/* query buffer size */
	cusolverSpCheck( cusolverSpXcsrperm_bufferSizeHost(cusolverHandle, 
		n, 
		n, 
		nnz, 
		MatDescr, 
		rowptr, 
		colind, 
		p, 
		p, 
		&bufferSizeInBytes),
		"cusolverSpXcsrperm_bufferSizeHost", __LINE__);

	/* allocate working space */
	pBuffer = (integer_t*) spike_malloc( ALIGN_INT, 1, bufferSizeInBytes);

	/* permute the matrix */
	cusolverSpCheck( cusolverSpXcsrpermHost(cusolverHandle,
		n,
		n,
		nnz,
		MatDescr,
		rowptr,
		colind,
		p,
		p, // actually, q for nonsymmetric
		map,
		pBuffer),
		"cusolverSpXcsrpermHost", __LINE__);


	spike_nullify(p);
	spike_nullify(pBuffer);

	return (SPIKE_SUCCESS);
};