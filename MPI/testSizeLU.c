
#include "spike_analysis.h"

int main(int argc, char *argv[]){
	MPI_Init( &argc, &argv );
	const integer_t nrhs = 1;
	get_maximum_av_host_memory();
	printf("Before Charge RS\n");

	matrix_t* A = matrix_LoadCSR("../Tests/spike/permuted.bsit");
	printf("Matrix A Loaded\n");
	sm_schedule_t* S = spike_solve_analysis( A, nrhs );

	printf("Loading reduced system\n");
	matrix_t* R = matrix_LoadCSR("ReducedSystem.bsit");
	printf("Reduced System loaded\n");

	block_t*  xr = block_CreateReducedRHS( S->p, S->ku, S->kl, nrhs );
	block_t* yr = block_CreateEmptyBlock( xr->n, xr->m, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );

	get_maximum_av_host_memory();
	printf("After Charge RS\n");

	DirectSolverHander_t *handler = directSolver_CreateHandler();
	directSolver_Configure(handler, S->max_nrhs );

	get_maximum_av_host_memory();
	printf("After initialize PARDISO\n");

	directSolver_Host_Solve ( R->n, R->nnz, xr->m, R->colind, R->rowptr, R->aij, yr->aij, xr->aij);

	get_maximum_av_host_memory();
	printf("Finished\n");
	MPI_Finalize();
	return 0;
}
