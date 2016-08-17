/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  Parallel spike with MPI
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:32:39
 *       Revision:  none
 *       Compiler:  mpiicc
 *
 *         Author:  Albert Coca Abelló
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */
#include "spike_blocking.h"
#include <mpi.h>

void blockingFi(sm_schedule_t* S, block_t* fi, block_t* yit, block_t* yib, integer_t nrhs, integer_t master, DirectSolverHander_t *handler)
{
	integer_t size, rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p = rank;
	const integer_t r0 = S->n[p];
	const integer_t rf = S->n[p+1];
	const integer_t COLBLOCKINGDIST = S->blockingDistance;
	block_t*  yi = block_CreateEmptyBlock( rf - r0, COLBLOCKINGDIST, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	// block_SetBandwidthValues( yi, S->ku[p], S->kl[p] );

	integer_t col;
	
	if ( nrhs <= COLBLOCKINGDIST ) {
		/* blocking buffer */
		block_t *yij = block_CreateEmptyBlock( rf - r0, nrhs, S->ku[p], S->kl[p], _RHS_BLOCK_, _WHOLE_SECTION_);

		block_InitializeToValue( yij, __zero  ); // TODO: optimize using memset

		/* Extract the fi sub-block */
		//Not necessary

		/* solve the system for the RHS value */
		directSolver_SolveForRHS ( handler, nrhs, yij->aij, fi->aij );

		/* extract the yijt and yijb and add it to vit and vib*/
		block_ExtractTip_blocking          ( yit, yij, 0, nrhs, _TOP_SECTION_, _COLMAJOR_ );
		block_ExtractTip_blocking          ( yib, yij, 0, nrhs, _BOTTOM_SECTION_, _COLMAJOR_ );

		/* clean up */
		// block_Deallocate (fij );
		block_Deallocate (yij );
	}
	else{

		/* blocking buffer */
		block_t *fij = block_CreateEmptyBlock( rf - r0, COLBLOCKINGDIST, S->ku[p], S->kl[p], _RHS_BLOCK_, _WHOLE_SECTION_);//fi part of fi
		block_t *yij = block_CreateEmptyBlock( rf - r0, COLBLOCKINGDIST, S->ku[p], S->kl[p], _RHS_BLOCK_, _WHOLE_SECTION_);

		for(col = 0; (col + COLBLOCKINGDIST) < nrhs; col += COLBLOCKINGDIST ) {

			block_InitializeToValue( yij, __zero  ); // TODO: optimize using memset

			/* solve the system for the RHS value */
			directSolver_SolveForRHS ( handler, COLBLOCKINGDIST, yij->aij, &fi->aij[col * (rf - r0)] );

			/* extract the yit tip using fi as buffer, then, add it to the reduced system RHS */
			block_ExtractTip_blocking          ( yit, yij, col, col + COLBLOCKINGDIST, _TOP_SECTION_,    _COLMAJOR_ );
			block_ExtractTip_blocking          ( yib, yij, col, col + COLBLOCKINGDIST, _BOTTOM_SECTION_, _COLMAJOR_ );
		}

		if ( col < nrhs ) {
			block_InitializeToValue( yi, __zero  ); // TODO: optimize using memset

			/* Extract the fi sub-block */
			//block_ExtractBlock_blocking ( fi, f, r0, rf, col, nrhs );

			/* solve the system for the RHS value */
			directSolver_SolveForRHS ( handler, nrhs - col , yij->aij, &fi->aij[col * (rf - r0)] );

			/* extract the yit tip using fi as buffer, then, add it to the reduced system RHS */
			block_ExtractTip_blocking          ( yit, yi, col, nrhs - col, _TOP_SECTION_, _COLMAJOR_ );
			block_ExtractTip_blocking          ( yib, yi, col, nrhs - col, _BOTTOM_SECTION_, _COLMAJOR_ );
		}

		/* clean up */
		block_Deallocate (fij );
		block_Deallocate (yij );
	}

	/* Extract the tips of the yi block */
	/* clean up */
	block_Deallocate (yi );
}

block_t* blockingBi(sm_schedule_t* S, matrix_t* BiTmp, block_t* Vit, block_t* Vib, integer_t master, DirectSolverHander_t *handler)
{
	integer_t size, rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p = rank;
	const integer_t r0 = S->n[p];
	const integer_t rf = S->n[p+1];
	const integer_t COLBLOCKINGDIST = S->blockingDistance;

	integer_t col;
	//matrix_Deallocate( BiTmp );
	
	if ( S->ku[p] < COLBLOCKINGDIST ) {
		/* blocking buffer */
		block_t* Vij = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
		block_t* Bij = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );

		block_InitializeToValue( Bij, __zero  ); // TODO: optimize using memset

		/* Extract the Bi sub-block */
		block_BuildBlockFromMatrix_blocking (BiTmp, Bij, 0, rf - r0, 0, S->ku[p], _V_BLOCK_ );
		/* solve Aij * Vi = Bi */
		directSolver_SolveForRHS( handler, S->ku[p], Vij->aij, Bij->aij );

		/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
		block_ExtractTip_blocking   ( Vit, Vij, 0, S->ku[p], _TOP_SECTION_, _ROWMAJOR_ );

		/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
		block_ExtractTip_blocking    ( Vib, Vij, 0, S->ku[p], _BOTTOM_SECTION_, _ROWMAJOR_ );

		/* clean up */

		block_Deallocate( Vij );
		block_Deallocate( Bij );
	}
	else{
		/* blocking buffer */
		block_t* Vij = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
		block_t* Bij = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );

		for(col = 0; (col + COLBLOCKINGDIST) < S->ku[p]; col += COLBLOCKINGDIST ) {
			block_InitializeToValue( Bij, __zero  ); // TODO: optimize using memset

			/* Extract the Bi sub-block */
			block_BuildBlockFromMatrix_blocking (BiTmp, Bij, 0, 0, col, col + COLBLOCKINGDIST, _V_BLOCK_ );
			/* solve Aij * Vi = Bi */
			directSolver_SolveForRHS( handler, COLBLOCKINGDIST, Vij->aij, Bij->aij );

			/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi   ( Vit, Vij, 0, COLBLOCKINGDIST, col, _TOP_SECTION_, _COLMAJOR_ );


			/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi    ( Vib, Vij, 0, COLBLOCKINGDIST, col, _BOTTOM_SECTION_, _COLMAJOR_ );
		}

		if ( col < S->ku[p] ) {
			/* blocking buffer */
			block_InitializeToValue( Bij, __zero  ); // TODO: optimize using memset
			block_InitializeToValue( Vij, __zero  ); // TODO: optimize using memset

			/* Extract the Bi sub-block */
			block_BuildBlockFromMatrix_blocking ( BiTmp, Bij, 0, 0, col, S->ku[p], _V_BLOCK_ );
			/* solve Aij * Vi = Bi */
			directSolver_SolveForRHS( handler, S->ku[p] - col, Vij->aij, Bij->aij );
			/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi   ( Vit, Vij, 0, S->ku[p] - col, col, _TOP_SECTION_, _COLMAJOR_ );

			/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi    ( Vib, Vij, 0, S->ku[p] - col, col, _BOTTOM_SECTION_, _COLMAJOR_ );
		}

		/* clean up */
		block_Deallocate( Vij );
		block_Deallocate( Bij );

		block_Transpose_blocking( Vit->aij, S->kl[p], S->ku[p] );
		block_Transpose_blocking( Vib->aij, S->ku[p], S->ku[p] );
	}
	
	block_t* Bib = block_BuildBlockFromMatrix(BiTmp, _V_BLOCK_, S->kl[p], S->kl[p], S->ku[p], S->kl[p]);

	return Bib;
}

block_t* blockingCi(sm_schedule_t* S, matrix_t* CiTmp, block_t* Wit, block_t* Wib, integer_t master, DirectSolverHander_t *handler){

	integer_t size, rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p = rank;
	const integer_t r0 = S->n[p];
	const integer_t rf = S->n[p+1];
	const integer_t COLBLOCKINGDIST = S->blockingDistance;

	integer_t col;
	//matrix_Deallocate( BiTmp );
	
	if ( S->ku[p] <= COLBLOCKINGDIST ) {
		/* blocking buffer */
		block_t* Wij = block_CreateEmptyBlock ( rf - r0, S->kl[p], S->ku[p], S->kl[p], _W_BLOCK_, _WHOLE_SECTION_ );
		block_t* Cij = block_CreateEmptyBlock ( rf - r0, S->kl[p], S->ku[p], S->kl[p], _W_BLOCK_, _WHOLE_SECTION_ );

		block_InitializeToValue( Cij, __zero  ); // TODO: optimize using memset

		/* Extract the Bi sub-block */
		block_BuildBlockFromMatrix_blocking (CiTmp, Cij, 0, rf - r0, 0, S->kl[p], _W_BLOCK_ );

		/* solve Aij * Vi = Bi */
		directSolver_SolveForRHS( handler, S->kl[p], Wij->aij, Cij->aij );

		/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
		block_ExtractTip_blocking   ( Wit, Wij, 0, S->kl[p], _TOP_SECTION_, _ROWMAJOR_ );

		/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
		block_ExtractTip_blocking    ( Wib, Wij, 0, S->kl[p], _BOTTOM_SECTION_, _ROWMAJOR_ );

		/* clean up */
		block_Deallocate( Wij );
		block_Deallocate( Cij );
	}
	else{
		/* blocking buffer */
		block_t* Wij = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, S->ku[p], S->kl[p], _W_BLOCK_, _WHOLE_SECTION_ );
		block_t* Cij = block_CreateEmptyBlock ( rf - r0, COLBLOCKINGDIST, S->ku[p], S->kl[p], _W_BLOCK_, _WHOLE_SECTION_ );

		for(col = 0; (col + COLBLOCKINGDIST) < S->kl[p]; col += COLBLOCKINGDIST ) {
			block_InitializeToValue( Cij, __zero  ); // TODO: optimize using memset

			/* Extract the Bi sub-block */
			block_BuildBlockFromMatrix_blocking (CiTmp, Cij, 0, 0, col, col + COLBLOCKINGDIST, _W_BLOCK_ );
			/* solve Aij * Vi = Bi */
			directSolver_SolveForRHS( handler, COLBLOCKINGDIST, Wij->aij, Cij->aij );

			/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi   ( Wit, Wij, 0, COLBLOCKINGDIST, col, _TOP_SECTION_, _COLMAJOR_ );


			/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi    ( Wib, Wij, 0, COLBLOCKINGDIST, col, _BOTTOM_SECTION_, _COLMAJOR_ );
		}

		if ( col < S->kl[p] ) {
			/* blocking buffer */
			block_InitializeToValue( Cij, __zero  ); // TODO: optimize using memset
			block_InitializeToValue( Wij, __zero  ); // TODO: optimize using memset

			/* Extract the Bi sub-block */
			block_BuildBlockFromMatrix_blocking ( CiTmp, Cij, 0, 0, col, S->kl[p], _W_BLOCK_ );

			/* solve Aij * Vi = Bi */
			directSolver_SolveForRHS( handler, S->kl[p] - col, Wij->aij, Cij->aij );
			/* extract the Vit tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi   ( Wit, Wij, 0, S->kl[p] - col, col, _TOP_SECTION_, _COLMAJOR_ );

			/* extract the Vib tip using Bi as buffer, then, add it to the reduced system */
			block_ExtractTip_blocking_mpi    ( Wib, Wij, 0, S->kl[p] - col, col, _BOTTOM_SECTION_, _COLMAJOR_ );
		}

		/* clean up */

		if (rank==1) block_Print( Wit, "Wit");
		if (rank==1) block_Print( Wib, "Wib");

		block_Deallocate( Wij );
		block_Deallocate( Cij );

		block_Transpose_blocking( Wit->aij, S->kl[p], S->kl[p] );
		block_Transpose_blocking( Wib->aij, S->ku[p], S->kl[p] );
	}
	
	block_t* Cit = block_BuildBlockFromMatrix(CiTmp, _W_BLOCK_, S->kl[p], S->kl[p], S->ku[p], S->kl[p]);
	
	return Cit;
}
