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
#include "spike_mpi.h"
#include <unistd.h>

/* -------------------------------------------------------------------- */
/* .. Error Check for MPI.
/* -------------------------------------------------------------------- */
static void MPI_CheckCall( int err )
{
	integer_t len;
	char eMsg[500];
	if (err != MPI_SUCCESS){
		MPI_Error_string(err, eMsg, &len);
		error(eMsg);
		abort();
	}
};


/*#########################################################################################################################

												BASIC SEND AND RECV FUNCTIONS

##########################################################################################################################*/

/* -----------------------------------------------------------------------
	- Send / Recv MATRIX
---------------------------------------------------------------------- */

/* -------------------------------------------------------------------- */
/* .. Function to send matrix_t to process p, p must recive 
/* .. this matrix with recvMatrix function.
/* -------------------------------------------------------------------- */
void sendMatrix (matrix_t *Aij, integer_t p){
	
	MPI_CheckCall(MPI_Send(Aij		   , 5,					   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->colind, Aij->nnz,			   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->rowptr, Aij->n+1,			   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->aij,    Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD));
};

/* -------------------------------------------------------------------- */
/* .. Function to send matrix_t to process p, p must recive 
/* .. this matrix with recvMatrix function.
/* -------------------------------------------------------------------- */
void IsendMatrix (matrix_t *Aij, integer_t p){
	
	MPI_Request request;
	MPI_CheckCall(MPI_Isend(&Aij->n,     5,			    		MPI_INT, 	    p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(Aij->colind, Aij->nnz,		    	MPI_INT, 	    p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(Aij->rowptr, Aij->n+1,				MPI_INT, 	    p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD, &request));

};

/* -------------------------------------------------------------------- */
/* .. Function to send packed matrix_t to process p, p must recive 
/* .. this matrix with recvMatrix function.
/* -------------------------------------------------------------------- */
void sendMatrixPacked (matrix_t *Aij, integer_t p, integer_t tag){

	MPI_Request request;	
	integer_t position = 0, i;
	integer_t buffSize = (Aij->n+1 + Aij->nnz + 4)*sizeof(integer_t) + Aij->nnz*sizeof(complex_t) + sizeof(uLong_t);
	char *buff = (char*) malloc(buffSize*sizeof(char));
	
	MPI_Pack(&Aij->n	, 1		  			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(&Aij->nnz	, 1		  			  ,_MPI_ULONG_	   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(&Aij->ku	, 3		  			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->colind, Aij->nnz			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->rowptr, Aij->n+1			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->aij	, Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_ , buff, buffSize, &position, MPI_COMM_WORLD);
	debug("Aij->n:%d, Aij->nnz:%llu, Aij->ku:%d, Aij->kl:%d, Aij->type:%d", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->type);

	MPI_Isend(buff, position, MPI_PACKED, p, tag, MPI_COMM_WORLD, &request);
};

/* -------------------------------------------------------------------- */
/* .. Function for recive matrix_t from process p, p must send 
/* .. this matrix with sendMatrix function.
/* -------------------------------------------------------------------- */
matrix_t* recvMatrix (integer_t p){
	
	int t[5];
	MPI_Status status;

	MPI_CheckCall(MPI_Recv(t, 5, MPI_INT, p, 0, MPI_COMM_WORLD, &status));

	matrix_t* Aij = matrix_CreateEmptyMatrix( t[0], t[1] );
	Aij->ku       = t[2];
	Aij->kl       = t[3];
	Aij->type     = t[4];
	
	MPI_CheckCall(MPI_Recv(Aij->colind, Aij->nnz, 	           MPI_INT,		   p, 0, MPI_COMM_WORLD, &status));
	MPI_CheckCall(MPI_Recv(Aij->rowptr, Aij->n+1, 	     	   MPI_INT, 	   p, 0, MPI_COMM_WORLD, &status));
	MPI_CheckCall(MPI_Recv(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD, &status));
	debug("Aij->n:%d, Aij->nnz:%llu, Aij->ku:%d, Aij->kl:%d, Aij->type:%d", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->type);

	return Aij;
}

/* -------------------------------------------------------------------- */
/* .. Function for recive matrix_t from process p, p must send 
/* .. this matrix with sendMatrix function.
/* -------------------------------------------------------------------- */
matrix_t* recvMatrixPacked (integer_t p, integer_t tag){
	
	integer_t buffSize = 0, position = 0, i;
	MPI_Status  status;
	integer_t t[3], n, nnz;

	MPI_Probe(p, tag, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_PACKED, &buffSize);
	char* buff = (char*)malloc(sizeof(char) * buffSize);

	MPI_Recv(buff, buffSize, MPI_PACKED, p, tag, MPI_COMM_WORLD, &status);
	MPI_Unpack(buff, buffSize, &position, &n,   1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, &nnz, 1, _MPI_ULONG_, MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, t,    3, MPI_INT, MPI_COMM_WORLD);

	matrix_t* Aij = matrix_CreateEmptyMatrix( n, nnz );
	Aij->ku = t[0];
	Aij->kl = t[1];
	Aij->type  = t[2];
	
	MPI_Unpack(buff, buffSize, &position, Aij->colind, Aij->nnz			   , MPI_INT	   , MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, Aij->rowptr, Aij->n+1			   , MPI_INT	   , MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, Aij->aij	 , Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_, MPI_COMM_WORLD);
	debug("Aij->n:%d, Aij->nnz:%llu, Aij->ku:%d, Aij->kl:%d, Aij->type:%d", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->type);
	return Aij;
}


/* -----------------------------------------------------------------------
	-- Send / Recv BLOCK
---------------------------------------------------------------------- */

/* -------------------------------------------------------------------- */
/* .. Function to send block_t to process p, p must recive 
/* .. this matrix with recvBlock function.
/* -------------------------------------------------------------------- */
void sendBlock (block_t *b, integer_t p){
	
	integer_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;

	MPI_CheckCall(MPI_Send(b, 		6		,   MPI_INT		  , p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(b->aij,  sendCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD));

}

/* -------------------------------------------------------------------- */
/* .. Function to send block_t to process p, p must recive 
/* .. this matrix with recvBlock function.
/* -------------------------------------------------------------------- */
void IsendBlock (block_t *b, integer_t p){
	
	integer_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;
	MPI_Request request;

	MPI_CheckCall(MPI_Isend(b, 6, MPI_INT, p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(b->aij,  sendCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD, &request));

}

/* -------------------------------------------------------------------- */
/* .. Function to send block_t to process p, p must recive 
/* .. this matrix with recvBlock function.
/* -------------------------------------------------------------------- */
void sendBlockPacked (block_t *b, integer_t p, integer_t tag){
	
	MPI_Request request;
	uLong_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;
	uLong_t buffSize = 6*sizeof(integer_t) + sendCount*sizeof(complex_t);
	debug("b->n:%d, b->m:%d, b->ku:%d, b->kl:%d", b->n, b->m, b->ku, b->kl);
	char* buff = (char*) spike_malloc(ALIGN_INT, buffSize, sizeof(char));
	
	integer_t position = 0;
	MPI_Pack(b	   , 6		  ,  MPI_INT	   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(b->aij, sendCount, _MPI_COMPLEX_T_, buff, buffSize, &position, MPI_COMM_WORLD);

	MPI_Isend(buff, buffSize/4, MPI_INT, p, tag, MPI_COMM_WORLD, &request);

}

/* -------------------------------------------------------------------- */
/* .. Function for recive block_t from process p, p must send 
/* .. this block with sendBlock function.
/* -------------------------------------------------------------------- */
block_t* recvBlock (integer_t p){

	integer_t t[6], recvCount;
	MPI_Status  status;
	
	MPI_CheckCall(MPI_Recv(t, 6, MPI_INT, p, 0, MPI_COMM_WORLD, &status));
	
	block_t *b = block_CreateEmptyBlock(t[2], t[3], t[4], t[5], t[0], t[1]);
	recvCount = (b->n)*(b->m)*_MPI_COUNT_;
	
	MPI_CheckCall(MPI_Recv(b->aij, recvCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD, &status));

	return b;
}

/* -------------------------------------------------------------------- */
/* .. Function for recive block_t from process p, p must send 
/* .. this block with sendBlock function.
/* -------------------------------------------------------------------- */
block_t* recvBlockPacked (integer_t p, integer_t tag){

	integer_t buffSize=0, position = 0;
	MPI_Status  status;
	integer_t t[6];

	MPI_Probe(p, tag, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_PACKED, &buffSize);
	char* buff = (char*) spike_malloc(ALIGN_INT, buffSize, sizeof(char));	

	MPI_Recv(buff, buffSize, MPI_INT, p, tag, MPI_COMM_WORLD, &status);

	MPI_Unpack(buff, buffSize, &position, t, 6, MPI_INT, MPI_COMM_WORLD);

	block_t *b = block_CreateEmptyBlock(t[2], t[3], t[4], t[5], t[0], t[1]);
	integer_t recvCount = (b->n)*(b->m)*_MPI_COUNT_;
	MPI_Unpack(buff, buffSize, &position, b->aij, recvCount , _MPI_COMPLEX_T_ , MPI_COMM_WORLD);
	debug("b->n:%d, b->m:%d, b->ku:%d, b->kl:%d", b->n, b->m, b->ku, b->kl);

	return b;
}

void sendSchedulePacked(sm_schedule_t* S, integer_t p){

	integer_t buffSize;
	buffSize = (5 + 2 + S->p*4)*sizeof(integer_t);
	char *buff = (char*) spike_malloc(ALIGN_INT, buffSize, sizeof(char));
	
	integer_t position = 0;
	MPI_Pack(S, 	5, 		MPI_INT, buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->n,  S->p+1, MPI_INT, buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->r, 	S->p+1, MPI_INT, buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->ku, S->p,   MPI_INT, buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->kl, S->p,   MPI_INT, buff, buffSize, &position, MPI_COMM_WORLD);
	
	MPI_Send(buff, position, MPI_PACKED, p, 0, MPI_COMM_WORLD);
	
}

sm_schedule_t* recvSchedulePacked(integer_t p){
	
	integer_t buffSize=0, position = 0, t[5];
	MPI_Status  status;
	sm_schedule_t* S = (sm_schedule_t*) spike_malloc(ALIGN_INT, 1, sizeof(sm_schedule_t));

	MPI_Probe(p, 0, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_PACKED, &buffSize);
	char* buff = (char*) spike_malloc(ALIGN_INT, buffSize, sizeof(char));
	
	MPI_Recv(buff, buffSize, MPI_PACKED, p, 0, MPI_COMM_WORLD, &status);

	MPI_Unpack(buff, buffSize, &position, t, 5, MPI_INT, MPI_COMM_WORLD);
	S->p        = t[0];
	S->max_n    = t[1];
	S->max_m    = t[2];
	S->max_nrhs = t[3];
	S->blockingDistance = t[4];

	S->n     = (integer_t*) spike_malloc(ALIGN_INT, S->p +1, sizeof(integer_t));
	S->r     = (integer_t*) spike_malloc(ALIGN_INT, S->p +1, sizeof(integer_t));
	S->ku    = (integer_t*) spike_malloc(ALIGN_INT, S->p, sizeof(integer_t));
	S->kl    = (integer_t*) spike_malloc(ALIGN_INT, S->p, sizeof(integer_t));

	MPI_Unpack(buff, buffSize, &position, S->n,  S->p+1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, S->r,  S->p+1, MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, S->ku, S->p  , MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, S->kl, S->p  , MPI_INT, MPI_COMM_WORLD);
	
	return S;
}

/*#########################################################################################################################

												COMPLEX SEND AND RECV FUNCTIONS

##########################################################################################################################*/

/* -------------------------------------------------------------------- 
	Send schedule to all nodes.
 -------------------------------------------------------------------- */
void scatterSchedule(sm_schedule_t* S){

	integer_t i, size;
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p;
	for(p=1; p < size; p++) sendSchedulePacked(S, p);
}

/* -------------------------------------------------------------------- 
	Asyncronous Send of Aij, Bi, Ci and fi to all nodes
 -------------------------------------------------------------------- */
void scatterAijBiCiFi(sm_schedule_t* S, matrix_t* A, block_t* f){

	integer_t i, size;
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p, first;
	for(p=1; p < size; p++)
	{
		integer_t r0, rf;
		if(MASTER_WORKING){
			r0 = S->n[p];
			rf = S->n[p+1];
			first = 0;
		}
		else{
			r0 = S->n[p-1];
			rf = S->n[p];
			first = 1;
		}

		matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf, _DIAG_BLOCK_);
		sendMatrixPacked(Aij, p, AIJ_TAG);

		block_t*  fi  = block_ExtractBlock( f, r0, rf );
		block_SetBandwidthValues( fi, A->ku, A->kl );
		sendBlockPacked(fi, p, FI_TAG);

		// Clean up
		matrix_Deallocate( Aij);
		block_Deallocate (fi );
		
		if(p == first){
			matrix_t* Bi = matrix_ExtractMatrix (A, rf - A->ku, rf, rf, rf + A->ku, _B_BLOCK_);
			sendMatrixPacked(Bi, p, BI_TAG);
			matrix_Deallocate( Bi );
		}

		else if (p == ( size -1)){
			matrix_t* Ci = matrix_ExtractMatrix (A, r0, r0 + A->kl, r0 - A->kl, r0 ,_C_BLOCK_);
			sendMatrixPacked(Ci, p, CI_TAG);
			matrix_Deallocate( Ci );
		}

		else{
			matrix_t* Bi = matrix_ExtractMatrix(A, rf - A->ku, rf, rf, rf + A->ku, _B_BLOCK_);
			matrix_t* Ci = matrix_ExtractMatrix(A, r0, r0 + A->kl, r0 - A->kl, r0 ,_C_BLOCK_);
			sendMatrixPacked(Bi, p, BI_TAG);				
			sendMatrixPacked(Ci, p, CI_TAG);
			matrix_Deallocate( Bi );
			matrix_Deallocate( Ci );
		}
	}
}

/* -------------------------------------------------------------------- 
	Asyncronous Recv of Aij, Bi, Ci and Fi from all nodes and
	prepare the Reduced System to be solved.
 -------------------------------------------------------------------- */
void gatherReducedSystem(sm_schedule_t* S, matrix_t* R, block_t* xr){

	integer_t i, size;
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Status  status;
	integer_t total_recv, offset;

	if(MASTER_WORKING){
		total_recv = 6*(size-1)-2;
		offset = 0;
	}
	else{
		total_recv = 6*(size-1)-4;
		offset = 1;
	}

	for(i=0; i<total_recv; i++){
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		switch(status.MPI_TAG) {
			case VIWI_TAG:
			{
				block_t* b = recvBlockPacked(status.MPI_SOURCE, VIWI_TAG);
				matrix_AddTipToReducedMatrix( S->p, status.MPI_SOURCE - offset, S->n, S->ku, S->kl, R, b);
				block_Deallocate(b);
				break;
			}
			case YI_TAG:
			{
				block_t* b = recvBlockPacked(status.MPI_SOURCE, YI_TAG);
				block_AddTipTOReducedRHS(status.MPI_SOURCE - offset, S->ku, S->kl, xr, b);
				block_Deallocate(b);
				break;
			}
		}
	}
}

/* -------------------------------------------------------------------- 
	Asyncronous Send of Aij, Bi, Ci and fi to all nodes
 -------------------------------------------------------------------- */
void scatterXiFi(sm_schedule_t* S, block_t* x, block_t* f, block_t* yr){
	
	integer_t i, size;
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p, first;
	for(p=1; p < size; p++){
		integer_t obs, obe, rbs, rbe, ni, ku, kl;
		if(MASTER_WORKING){
			/* compute the limits of the blocks */
			obs = S->n[p];        		/* original system starting row */
			obe = S->n[p+1];	  		/* original system ending row   */
			rbs = S->r[p];		  		/* reduceed system starting row */
			rbe = S->r[p+1];			/* reduced system ending row    */
			ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */
			first = 0;
			ku = S->ku[p];
			kl = S->kl[p];
		}
		else{
			/* compute the limits of the blocks */
			obs = S->n[p-1];        		/* original system starting row */
			obe = S->n[p];	  		/* original system ending row   */
			rbs = S->r[p-1];		  		/* reduceed system starting row */
			rbe = S->r[p];			/* reduced system ending row    */
			ni  = S->n[p] - S->n[p]; 	/* number of rows in the block  */
			first = 1;
			ku = S->ku[p-1];
			kl = S->kl[p-1];
		}

		/* extract xi sub-block */
		block_t*  xi  = block_ExtractBlock(x, obs, obe );
		sendBlockPacked(xi, p, XI_TAG);

		/* extract fi sub-block */
		block_t*  fi  = block_ExtractBlock(f, obs, obe );
		sendBlockPacked(fi, p, FI_TAG);
		
		if ( p == first ){

			block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + ku);
			sendBlockPacked(xt_next, p, XT_NEXT_TAG);
			block_Deallocate (xt_next);
		}

		else if ( p == ( size -1)){

			block_t* xb_prev = block_ExtractBlock ( yr, rbs - kl, rbs );
			sendBlockPacked(xb_prev, p, XT_PREV_TAG);
			block_Deallocate (xb_prev);
		}

		else{
			block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + ku);
			sendBlockPacked(xt_next, p, XT_NEXT_TAG);
			block_Deallocate (xt_next);
			
			block_t* xb_prev = block_ExtractBlock ( yr, rbs - kl, rbs );
			sendBlockPacked(xb_prev, p, XT_PREV_TAG);
			block_Deallocate (xb_prev);
		}
		block_Deallocate( fi );
		block_Deallocate( xi );
	}
}

void gatherXi(sm_schedule_t* S, block_t* x){
	integer_t i, size;
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p, offset;
	MPI_Status  status;
	if(MASTER_WORKING) offset = 0;
	else offset = 1;
	for(p=1; p < size; p++){
		MPI_Probe(MPI_ANY_SOURCE, XI_TAG, MPI_COMM_WORLD, &status);
		block_t* xi = recvBlockPacked(status.MPI_SOURCE, XI_TAG);
		block_AddBlockToRHS(x, xi, S->n[status.MPI_SOURCE-offset], S->n[status.MPI_SOURCE+1-offset]);
		block_Deallocate ( xi );
	}
}

void workerSolveAndSendTips(sm_schedule_t* S, integer_t master, integer_t nrhs, matrix_t* Aij, block_t *Bib, block_t *Cit, DirectSolverHander_t *handler){

	block_t* Bib2;
	block_t* Cit2;
	
	integer_t rank, size, i, max_work, p;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Status  status;
	if(MASTER_WORKING){
		if(rank == 0 || rank == size-1) max_work = 2;
		else max_work = 3;
		p = rank;
	}
	else{
		if(rank == 1 || rank == size-1) max_work = 2;
		else max_work = 3;
		p = rank - 1;
	}
	const integer_t r0 = S->n[p];
	const integer_t rf = S->n[p+1];

	debug("Max Work %d", max_work);
	for(i=0; i<max_work; i+=1){
		MPI_Probe(master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		printf("MPI_STATUS.TAG: %d, rank: %d, FI_TAG: %d\n", status.MPI_TAG, rank, CI_TAG);
		switch(status.MPI_TAG) {
			case FI_TAG:
			{
				/* solve the system for the RHS value */
				block_t*  fi = recvBlockPacked(master, FI_TAG);
				block_t* yit;
				block_t* yib;

				if(!BLOCKING){
					block_t*  yi = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
					block_SetBandwidthValues( yi, S->ku[p], S->kl[p] );

					/* Solve Ai * yi = fi */
					directSolver_SolveForRHS( handler, nrhs, yi->aij, fi->aij);
					yit = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
					yib = block_ExtractTip( yi, _BOTTOM_SECTION_, _COLMAJOR_ );
					block_Deallocate (yi );
				}
				else{
	
					yit = block_CreateEmptyBlock( S->kl[p], nrhs, S->ku[p], S->kl[p], _RHS_BLOCK_, _TOP_SECTION_ );
					yib = block_CreateEmptyBlock( S->ku[p], nrhs, S->ku[p], S->kl[p], _RHS_BLOCK_, _BOTTOM_SECTION_ );
					blockingFi(S, fi, yit, yib, nrhs, master, p, handler);
				}

				sendBlockPacked(yit, master, YI_TAG);
				sendBlockPacked(yib, master, YI_TAG);

				/* clean up */
				block_Deallocate (fi);
				block_Deallocate (yit);
				block_Deallocate (yib);
				
				break;
			}

			case BI_TAG:
			{
				matrix_t* BiTmp = recvMatrixPacked(master, BI_TAG);
				block_t* Vit;
				block_t* Vib;

				if(!BLOCKING){
					block_t* Vi = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
					block_t* Bi = block_BuildBlockFromMatrix(BiTmp, _V_BLOCK_, Aij->n, Aij->n, S->ku[p], S->kl[p]);

					/* solve Ai * Vi = Bi */
					directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij);

					Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
					Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
					Bib2 = block_ExtractTip( Bi, _BOTTOM_SECTION_, _COLMAJOR_ );
					block_Deallocate( Bi);
					block_Deallocate( Vi);
				}

				else{

					Vit = block_CreateEmptyBlock( S->kl[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _TOP_SECTION_ );
					Vib = block_CreateEmptyBlock( S->ku[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _BOTTOM_SECTION_ );
					Bib2 = blockingBi(S, BiTmp, Vit, Vib, master, p, handler);
				}

				sendBlockPacked(Vit, master, VIWI_TAG);
				sendBlockPacked(Vib, master, VIWI_TAG);

				memcpy(Bib->aij, Bib2->aij, Bib->n*Bib->m*sizeof(complex_t));

				block_Deallocate( Vit);
				block_Deallocate( Vib);

				break;
			}

			case CI_TAG:
			{
				matrix_t* CiTmp = recvMatrixPacked(master, CI_TAG);
				block_t* Wit;
				block_t* Wib;
				if(!BLOCKING){
					block_t* Wi = block_CreateEmptyBlock( rf - r0, S->kl[p], S->ku[p], S->kl[p], _W_BLOCK_, _WHOLE_SECTION_ );
					block_t* Ci = block_BuildBlockFromMatrix(CiTmp, _W_BLOCK_, Aij->n, Aij->n, S->ku[p], S->kl[p]);

					/* solve Ai  * Wi = Ci */
					directSolver_SolveForRHS( handler, Wi->m, Wi->aij, Ci->aij);
		
					Wit = block_ExtractTip( Wi, _TOP_SECTION_, _ROWMAJOR_ );
					Wib = block_ExtractTip( Wi, _BOTTOM_SECTION_, _ROWMAJOR_ );
					Cit2 = block_ExtractTip(Ci, _TOP_SECTION_, _COLMAJOR_ );					
					block_Deallocate( Ci );
					block_Deallocate( Wi );					
				}
				else{
					Wit = block_CreateEmptyBlock( S->kl[p], S->kl[p], S->ku[p], S->kl[p], _W_BLOCK_, _TOP_SECTION_ );
					Wib = block_CreateEmptyBlock( S->ku[p], S->kl[p], S->ku[p], S->kl[p], _W_BLOCK_, _BOTTOM_SECTION_ );
					Cit2 = blockingCi(S, CiTmp, Wit, Wib, master, p, handler);
				}

				sendBlockPacked(Wit, master, VIWI_TAG);
				sendBlockPacked(Wib, master, VIWI_TAG);

				memcpy(Cit->aij, Cit2->aij, Cit->n*Cit->m*sizeof(complex_t));
				block_Print(Cit, "Cit");
	
				block_Deallocate( Wit);
				block_Deallocate( Wib);
			
				break;
			}
		}
	}
}

void workerSolveBackward(sm_schedule_t* S, block_t* Bib, block_t* Cit, integer_t master, DirectSolverHander_t *handler){

	integer_t max_work, i, p, rank, size;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Status  status;

	if(MASTER_WORKING){
		p = rank;
		if(rank == 0 || rank == size-1) max_work = 1;
		else max_work = 2;
	}
	else{
		p = rank - 1;
		if(rank == 1 || rank == size-1) max_work = 1;
		else max_work = 2;
	}
	const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */

	block_t* xi = recvBlockPacked(master, XI_TAG);
	block_t* fi = recvBlockPacked(master, FI_TAG);

	for(i=0; i<max_work; i++){
		MPI_Probe(master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		switch(status.MPI_TAG) {
			case XT_NEXT_TAG:
			{
				block_t* xt_next = recvBlockPacked(master, XT_NEXT_TAG);
		
				/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
				gemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
					Bib->n,    						/* m - number of rows of A    */
					xt_next->m, 					/* n - number of columns of B */
					Bib->m,    						/* k - number of columns of A */
					__nunit,						/* alpha                      */
					Bib->aij, 						/* A block                    */
					Bib->n,    						/* lda - first dimension of A */
					xt_next->aij, 					/* B block                    */
					xt_next->n,    					/* ldb - first dimension of B */
					__punit,						/* beta                       */
					&fi->aij[ni - S->ku[p]], 		/* C block                    */
					ni ); 					 		/* ldc - first dimension of C */
		
				/* Solve Ai * xi = fi */
				directSolver_SolveForRHS(handler, xi->m, xi->aij, fi->aij);

				block_Deallocate ( Bib );
				block_Deallocate ( xt_next);

				break;
			}
			case XT_PREV_TAG:
			{
				block_t* xb_prev = recvBlockPacked(master, XT_PREV_TAG);

				/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */ 
				gemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
					Cit->n,    						/* m - number of rows of A    */
					xb_prev->m, 					/* n - number of columns of B */
					Cit->m,    						/* k - number of columns of A */
					__nunit,						/* alpha                      */
					Cit->aij, 						/* A block                    */
					Cit->n,    						/* lda - first dimension of A */
					xb_prev->aij, 					/* B block                    */
					xb_prev->n,    					/* ldb - first dimension of B */
					__punit,						/* beta                       */
					fi->aij, 			 		    /* C block                    */
					ni );		 					/* ldc - first dimension of C */

				/* Solve Ai * xi = fi */
				directSolver_SolveForRHS(handler, xi->m, xi->aij, fi->aij);

				block_Deallocate ( Cit );
				block_Deallocate ( xb_prev);
				break;
			}
		}
	}
	sendBlockPacked(xi, master, XI_TAG);
	block_Deallocate (xi);
	block_Deallocate (fi);
}

void masterWorkFactorize(DirectSolverHander_t *handler, sm_schedule_t* S, matrix_t* A, block_t* f, matrix_t* R, block_t* Bib, block_t* xr, integer_t nrhs){

	integer_t rank, size;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p = 0;
	const integer_t r0 = S->n[p];
	const integer_t rf = S->n[p+1];
	matrix_t* Aij = matrix_ExtractMatrix(A, r0, rf, r0, rf, _DIAG_BLOCK_);
	block_t* Bib2;

	directSolver_Factorize( handler,
		Aij->n,
		Aij->nnz,
		Aij->colind,
		Aij->rowptr,
		Aij->aij);

	/* solve the system for the RHS value */
	block_t*  fi = block_ExtractBlock( f, r0, rf );
	block_t*  yi = block_CreateEmptyBlock( rf - r0, nrhs, 0, 0, _RHS_BLOCK_, _WHOLE_SECTION_ );
	block_SetBandwidthValues( yi, S->ku[p], S->kl[p] );
	block_SetBandwidthValues( fi, S->ku[p], S->kl[p] );

	/* Solve Ai * yi = fi */
	directSolver_SolveForRHS( handler, nrhs, yi->aij, fi->aij);

	/* Extract the tips of the yi block */
	block_t* yit = block_ExtractTip( yi, _TOP_SECTION_   , _COLMAJOR_ );
	block_t* yib = block_ExtractTip( yi, _BOTTOM_SECTION_, _COLMAJOR_ );
	block_AddTipTOReducedRHS(rank, S->ku, S->kl, xr, yit);
	block_AddTipTOReducedRHS(rank, S->ku, S->kl, xr, yib);

	/* clean up */
	block_Deallocate (yi );
	block_Deallocate (yit);
	block_Deallocate (yib);

	block_t* Vi = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
	block_t* Bi = matrix_ExtractBlock    ( A, r0, rf, rf, rf + A->ku, _V_BLOCK_ );

	/* solve Ai * Vi = Bi */
	directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij);

	block_t* Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
	block_t* Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
	matrix_AddTipToReducedMatrix( S->p, rank, S->n, S->ku, S->kl, R, Vit);
	matrix_AddTipToReducedMatrix( S->p, rank, S->n, S->ku, S->kl, R, Vib);

	Bib2 = block_ExtractTip( Bi, _BOTTOM_SECTION_, _COLMAJOR_ );
	memcpy(Bib->aij, Bib2->aij, Bib->n*Bib->m*sizeof(complex_t));

	block_Deallocate( Bi );
	block_Deallocate( Vi);
	block_Deallocate( Vit);
	block_Deallocate( Vib);
	block_Deallocate( fi);
	block_Deallocate (Bib2);
}

void masterWorkBackward(sm_schedule_t* S, block_t* yr, block_t* f, block_t* x, block_t* Bib, DirectSolverHander_t *handler){

	integer_t rank, size;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p = rank; 
	const integer_t obs = S->n[p];        		/* original system starting row */
	const integer_t obe = S->n[p+1];	  		/* original system ending row   */
	const integer_t rbs = S->r[p];		  		/* reduceed system starting row */
	const integer_t rbe = S->r[p+1];			/* reduced system ending row    */
	const integer_t ni  = S->n[p+1] - S->n[p]; 	/* number of rows in the block  */

	block_t* xt_next = block_ExtractBlock ( yr, rbe, rbe + S->ku[p]);
	block_t*  xi  = block_ExtractBlock(x, obs, obe );
	block_t*  fi2  = block_ExtractBlock(f, obs, obe );

	/* Backward substitution, implicit scheme: xi = -1.0 * Bi * xit  + fi */
	gemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
		Bib->n,    						/* m - number of rows of A    */
		xt_next->m, 					/* n - number of columns of B */
		Bib->m,    						/* k - number of columns of A */
		__nunit,						/* alpha                      */
		Bib->aij, 						/* A block                    */
		Bib->n,    						/* lda - first dimension of A */
		xt_next->aij, 					/* B block                    */
		xt_next->n,    					/* ldb - first dimension of B */
		__punit,						/* beta                       */
		&fi2->aij[ni - S->ku[p]], 		/* C block                    */
		ni ); 					 		/* ldc - first dimension of C */

	/* Solve Ai * xi = fi */
	directSolver_SolveForRHS(handler, xi->m, xi->aij, fi2->aij);
	// directSolver_Host_Solve ( Aij->n, Aij->nnz, fi2->m, Aij->colind, Aij->rowptr, Aij->aij, xi->aij, fi2->aij);
	block_AddBlockToRHS(x, xi, S->n[rank], S->n[rank+1]);

	block_Deallocate(xi);
	block_Deallocate(fi2);
	block_Deallocate(xt_next);
}

/* -------------------------------------------------------------------- */
/* .. Debug function for MPI. 
/* -------------------------------------------------------------------- */
void my_debug(char *file, integer_t line, const char *func, char *type, const char *fmt, ...){

	FILE *fp;
	integer_t rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	char fpName[20];
	sprintf(fpName, "Mpi_Debug%d.log", rank);
	fp = fopen (fpName, "a");

	time_t timer;
    char buffer[26];
	struct tm* tm_info;
	time(&timer);
	tm_info = localtime(&timer);
	strftime(buffer, 26, "%H:%M:%S", tm_info);

	va_list args;
	va_start(args, fmt);
	fprintf(fp, "%s %s:%d:%s::%s ", type, file, line, func, buffer);
	vfprintf(fp, fmt, args);
	fprintf(fp, "\n");
	va_end(args);
	
	fclose(fp);
}
