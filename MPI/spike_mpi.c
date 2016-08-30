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

/** 
 *	@file spike_mpi.c 
 *  @brief Distributed memory spike library.
 */
#include "spike_mpi.h"
#define MS 50

MPI_Request	arrayRequest[MS]; /**< Array of requests for asyncronous MPI functions */
char* arrayBuffer[MS];		  /**< Pointers to buffers to allocate when asyncronous MPI functions need it */
integer_t isFree[MS];		  /**< Bool array: 1 arrayBuffer has been free, 0 no. */
integer_t reqCount = 0;		  /**< Maximum allocated buffers for asyncronous MPI. */


/**
 * @brief 		Get the position of the first avaiable buffer.

				This function search for the first avaiable postion on the buffers
				to send asyncronous messages with MPI.

 * @return		Index of the first avaiable buffer. 
 */
//TODO Optimize the acces.
integer_t findAvailReq(){
	integer_t i;
	integer_t flag;
	for(i=0; i<reqCount; i++){
		MPI_Request_get_status(arrayRequest[i], &flag, MPI_STATUS_IGNORE);
		if(flag){
			if(!isFree[i]){
					free(arrayBuffer[i]);
				}
			else isFree[i] = 0;
			return i;
		}
	}
	reqCount++;
	return reqCount-1;
}

/**
 * @brief 		Free Isend buffers.

				This function check all the MPI buffers for asyncronous send
				and if the send data is delivered then it will free the memory.

 * @param err	MPI error code.
 * @return 		Number of nonfree buffers.
 */
integer_t checkAndFreeRequest(){

	integer_t i;
	integer_t flag = 0;
	integer_t notFree = 0;
	get_maximum_av_host_memory();
	for(i=0; i<reqCount; i++){
		MPI_Request_get_status(arrayRequest[i], &flag, MPI_STATUS_IGNORE);
		if(flag && !isFree[i]){
			isFree[i] = 1;
			free(arrayBuffer[i]);
		}
		else if(!flag)notFree++;
	}
	get_maximum_av_host_memory();
	return notFree;
}

/**
 * @brief 		Print the error string to the logfile.

 * @param err	MPI error code.
 */
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

/**
 * @brief 		Send Matrix Aij to process p, p must recive 
 * 				this matrix with recvMatrix function.
 * @param Aij 	Matrix to send.
 * @param p 	Destination.
 */
void sendMatrix (matrix_t *Aij, integer_t p){
	
	MPI_CheckCall(MPI_Send(Aij		   , 5,					   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->colind, Aij->nnz,			   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->rowptr, Aij->n+1,			   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->aij,    Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD));
};

/**
 * @brief 		Asyncronous send of Matrix Aij to process p, p must recive 
 * 				this matrix with recvMatrix function.
 * @param Aij 	Matrix to send.
 * @param p 	Destination.
 */
void IsendMatrix (matrix_t *Aij, integer_t p){
	
	MPI_Request request;
	MPI_CheckCall(MPI_Isend(&Aij->n,     5,			    		MPI_INT, 	    p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(Aij->colind, Aij->nnz,		    	MPI_INT, 	    p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(Aij->rowptr, Aij->n+1,				MPI_INT, 	    p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD, &request));

};

/**
 * @brief 		Asyncronous send of Matrix Aij to process p, p must recive 
 * 				this matrix with recvMatrixPacked function.
 * @param Aij 	Matrix to send.
 * @param p 	Destination.
 * @param tag 	Tag of the message, we will need it later to recive it asyncronous.
 */
void sendMatrixPacked (matrix_t *Aij, integer_t p, integer_t tag){

	checkAndFreeRequest();
	int req = findAvailReq();
	integer_t position = 0, i;
	integer_t buffSize = (Aij->n+1 + Aij->nnz + 4)*sizeof(integer_t) + Aij->nnz*sizeof(complex_t) + sizeof(uLong_t);

	arrayBuffer[req] = (char*) malloc(buffSize*sizeof(char));
	char *buff = arrayBuffer[req];
	
	MPI_Pack(&Aij->n	, 1		  			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(&Aij->nnz	, 1		  			  ,_MPI_ULONG_	   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(&Aij->ku	, 3		  			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->colind, Aij->nnz			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->rowptr, Aij->n+1			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->aij	, Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_ , buff, buffSize, &position, MPI_COMM_WORLD);
	debug("Aij->n:%d, Aij->nnz:%llu, Aij->ku:%d, Aij->kl:%d, Aij->type:%d", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->type);

	MPI_Isend(buff, position, MPI_PACKED, p, tag, MPI_COMM_WORLD, &arrayRequest[req]);
	
};

/**
 * @brief 		Recive Matrix from process p, p must send 
 * 				this matrix from sendMatrix function.
 * @param p 	From witch process.
 * @return		Recived Matrix.
 */
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

/**
 * @brief 		Recive Matrix from process p, p must send 
 * 				this matrix from sendMatrixPacked function.
 * @param p 	From witch process.
 * @param tag 	Tag of the message, we will need it later to recive it asyncronous.
 * @return		Recived Matrix.
 */
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
	free(buff);
	return Aij;
}


/* -----------------------------------------------------------------------
	-- Send / Recv BLOCK
---------------------------------------------------------------------- */

/**
 * @brief 		Send block b to process p, p must recive 
 * 				this matrix with recvBlock function.
 * @param b 	Block to send.
 * @param p 	Destination.
 */
void sendBlock (block_t *b, integer_t p){
	
	integer_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;

	MPI_CheckCall(MPI_Send(b, 		6		,   MPI_INT		  , p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(b->aij,  sendCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD));

}

/**
 * @brief 		Asyncronous Send block b to process p, p must recive 
 * 				this matrix with recvBlock function.
 * @param b 	Block to send.
 * @param p 	Destination.
 */
void IsendBlock (block_t *b, integer_t p){
	
	integer_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;
	MPI_Request request;

	MPI_CheckCall(MPI_Isend(b, 6, MPI_INT, p, 0, MPI_COMM_WORLD, &request));
	MPI_CheckCall(MPI_Isend(b->aij,  sendCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD, &request));

}

/**
 * @brief 		Send block b to process p, p must recive 
 * 				this matrix with recvBlockPacked function.
 *				This function can send more than 2³¹ bytes.

 * @param b 	Block to send.
 * @param p 	Destination.
 * @param tag 	Tag of the message, we will need it later to recive it asyncronous.
 */
void sendBlockPacked (block_t *b, integer_t p, integer_t tag){
	
	MPI_Datatype BIG;
	MPI_Type_contiguous( BIG_MSG_SIZE, MPI_PACKED, &BIG );
	MPI_Type_commit(&BIG);

	checkAndFreeRequest();
	int req = findAvailReq();

	uLong_t sendCount = ((uLong_t)b->n)*((uLong_t)b->m)*(uLong_t)sizeof(complex_t) ;

	integer_t bigSize = (sendCount / BIG_MSG_SIZE)+1;
	uLong_t total = (uLong_t)bigSize*(uLong_t)BIG_MSG_SIZE;

	arrayBuffer[req] = (char*) malloc (total*sizeof(char) + 6*sizeof(integer_t));
	//char *buff = (char*) malloc (total*sizeof(char) + 6*sizeof(integer_t));

	integer_t position = 0;
	memcpy (&arrayBuffer[req][position], b, 6*sizeof(integer_t) );
	position = 6*sizeof(integer_t);

	memcpy (&arrayBuffer[req][position], b->aij, sendCount*sizeof(char));

	MPI_Isend(arrayBuffer[req], bigSize, BIG, p, tag, MPI_COMM_WORLD, &arrayRequest[req]);

	debug("b->n:%d, b->m:%d, b->ku:%d, b->kl:%d, buff: %llu", b->n, b->m, b->ku, b->kl, (uLong_t)bigSize*(uLong_t)BIG_MSG_SIZE);

}

/**
 * @brief 		Recive Block from process p, p must send 
 * 				this matrix from sendBlock function.
 * @param p 	From witch process.
 * @return		Recived Block.
 */
block_t* recvBlock (integer_t p){

	integer_t t[6], recvCount;
	MPI_Status  status;
	
	MPI_CheckCall(MPI_Recv(t, 6, MPI_INT, p, 0, MPI_COMM_WORLD, &status));
	
	block_t *b = block_CreateEmptyBlock(t[2], t[3], t[4], t[5], t[0], t[1]);
	recvCount = (b->n)*(b->m)*_MPI_COUNT_;
	
	MPI_CheckCall(MPI_Recv(b->aij, recvCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD, &status));

	return b;
}

/**
 * @brief 		Recive Block from process p, p must send 
 * 				this matrix from sendBlockPacked function.
 * @param p 	From witch process.
 * @param tag 	Tag of the message, we will need it later to recive it asyncronous.
 * @return		Recived Block.
 */
block_t* recvBlockPacked (integer_t p, integer_t tag){

	MPI_Datatype BIG;
	MPI_Type_contiguous( BIG_MSG_SIZE, MPI_PACKED, &BIG );
	MPI_Type_commit(&BIG);

	integer_t position = 0;
	integer_t bigSize = 0;
	uLong_t buffSize = 0;
	MPI_Status  status;
	integer_t t[6];

	MPI_Probe(p, tag, MPI_COMM_WORLD, &status);
	integer_t err = MPI_Get_count(&status, BIG, &bigSize);
	//debug("err:%d, MPI_UNDEFINED: %d", err, MPI_UNDEFINED);
	debug("BigSize: %d", bigSize);

	uLong_t total = (uLong_t)bigSize*(uLong_t)BIG_MSG_SIZE;

	debug("BuffSize: %llu", total);

	char* buff = (char*) spike_malloc(ALIGN_INT, total, sizeof(char));
	
	MPI_Recv(buff, bigSize, BIG, p, tag, MPI_COMM_WORLD, &status);

	memcpy (t, buff, 6*sizeof(integer_t) );

	//MPI_Unpack(buff, total, &position, t, 6, MPI_INT, MPI_COMM_WORLD);
	debug("t: %d, %d, %d, %d, %d, %d", t[0], t[1], t[2], t[3], t[4], t[5]);

	block_t *b = block_CreateEmptyBlock(t[2], t[3], t[4], t[5], t[0], t[1]);
	//integer_t recvCount = (b->n)*(b->m)*_MPI_COUNT_;

	//MPI_Unpack(buff, bigSize, &position, b->aij, recvCount , _MPI_COMPLEX_T_ , MPI_COMM_WORLD);
	uLong_t sendCount = ((uLong_t)b->n)*((uLong_t)b->m)*(uLong_t)sizeof(complex_t);
	memcpy (b->aij, &buff[6*sizeof(integer_t)], sendCount*sizeof(char) );
	

	debug("b->n:%d, b->m:%d, b->ku:%d, b->kl:%d, buffSize: %d", b->n, b->m, b->ku, b->kl, buffSize);
	spike_free(buff);

	return b;
}

/**
 * @brief 		Send schedule to process p, p must recive 
 * 				this schedule with recvSchedulePacked function.
 * @param p 	Destination.
 */
void sendSchedulePacked(dm_schedule_t* S, integer_t p){

	checkAndFreeRequest();
	int req = findAvailReq();
	integer_t buffSize;
	buffSize = (5 + 2 + S->p*4)*sizeof(integer_t);
	
	arrayBuffer[req] = (char*) malloc (buffSize*sizeof(char));
	
	integer_t position = 0;
	MPI_Pack(S, 	5, 		MPI_INT, arrayBuffer[req], buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->n,  S->p+1, MPI_INT, arrayBuffer[req], buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->r, 	S->p+1, MPI_INT, arrayBuffer[req], buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->ku, S->p,   MPI_INT, arrayBuffer[req], buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(S->kl, S->p,   MPI_INT, arrayBuffer[req], buffSize, &position, MPI_COMM_WORLD);
	
	MPI_Send(arrayBuffer[req], position, MPI_PACKED, p, 0, MPI_COMM_WORLD);
	
}

/**
 * @brief 		Recive schedule from process p, p must send 
 * 				this schedule from sendSchedulePacked function.
 * @param p 	From witch process.
 * @return		Recived Schedule.
 */
dm_schedule_t* recvSchedulePacked(integer_t p){
	
	integer_t buffSize=0, position = 0, t[5];
	MPI_Status  status;
	dm_schedule_t* S = (dm_schedule_t*) spike_malloc(ALIGN_INT, 1, sizeof(dm_schedule_t));

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

	spike_free(buff);
	
	return S;
}

/*#########################################################################################################################
												COMPLEX SEND AND RECV FUNCTIONS
##########################################################################################################################*/


/**
 * @brief 		Send schedule to all nodes.
 * @param S		Schedule to send.
 */
void scatterSchedule(dm_schedule_t* S){

	integer_t i, size;
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	integer_t p;
	for(p=1; p < size; p++) sendSchedulePacked(S, p);
}

/**
 * @brief 		Asyncronous send matrix Aij, Bi, Ci and block fi to all nodes

				Its only needed to send Bi to the first process and Ci to the last process,
				it will send Ci and Bi to all others process. It change when the Master
				process is working because its supose to be the first partiton, in this case
				the process number 1 (master = 0) will recive Bi and Ci.

				This function use sendMatrixPacked and sendBlockPacked that are Asyncronous.
 * @param S		Schedule of spike.
 * @param A		Original spike matrix.
 * @param f		Number of right hand sides of the spike system.
 */
void scatterAijBiCiFi(dm_schedule_t* S, matrix_t* A, block_t* f){

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

/**
 * @brief 		Gather all the tips and insert them to the Reduced System.

				This functions is an Asycronous function to recive all the tips,
				depending on the tag it will insert the tips on the Reduced System (R)
				or on the solution (xr).
								
				This function use recvBlockPacked dynamicaly to recive Asyncronous.
 * @param S		Schedule of spike.
 * @param R		Reduced System (empty).
 * @param xr	Solution of the reduced system (empty).
 */
void gatherReducedSystem(dm_schedule_t* S, matrix_t* R, block_t* xr){

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

/**
 * @brief 		Asyncronous send matrix xi, fi and fi to all nodes.

				Its only needed to send xt_next to the first process and xb_prev to the last process,
				it will send xt_next and xb_prev to all others process. It change when the Master
				process is working because its supose to be the first partiton, in this case
				the process number 1 (master = 0) will recive xt_next and xb_prev.

				This function use sendBlockPacked that is Asyncronous.
 * @param S		Schedule of spike.
 * @param f		f after solving Reduced System.
 * @param yr	yr after solving Reduced System.
 */

void scatterXiFi(dm_schedule_t* S, block_t* x, block_t* f, block_t* yr){
	
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

/**
 * @brief 		Asyncronous recive of all xi blocks from the other process and add it 
				to the final solution vector.
				This function use recvBlockPacked that is Asyncronous.

 * @param S		Spike Schedule.
 * @param x		block xi part of the final solution.
 */
void gatherXi(dm_schedule_t* S, block_t* x){
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


/**
 * @brief 		Worker recive Bi, Ci and fi Asyncronous and send the solution
				to master.

				Before this function you need to factorize Aij and during the factorization 
				the master is sending the blocks Bi, Ci, and fi, and it recive this blocks 
				in asyncronous way. When this function recive a block (Bi, Ci or fi) it 
				process this block and send to the master the top and the bottom part of the
				solution.

 * @param S			Spike Schedule.
 * @param master	Master node id.
 * @param nrhs		Number of right hand sides of the spike system.
 * @param Aij		Factorized Aij Matrix
 * @param Bib		Store the bottom part of Bi (it will be needed later).
 * @param Cit		Store the top part of Ci (it will be needed later).
 * @param handler 	Direct solver.
 */
void workerSolveAndSendTips(dm_schedule_t* S, integer_t master, integer_t nrhs, matrix_t* Aij, block_t *Bib, block_t *Cit, DirectSolverHander_t *handler){

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
		debug("MPI_STATUS.TAG: %d", status.MPI_TAG, rank);
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
					blockingFi(S, fi, yit, yib, nrhs, p, handler);
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
				debug("Recive Matrix Done");
				block_t* Vit;
				block_t* Vib;

				if(!BLOCKING){
					block_t* Vi = block_CreateEmptyBlock ( rf - r0, S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _WHOLE_SECTION_ );
					block_t* Bi = block_BuildBlockFromMatrix(BiTmp, _V_BLOCK_, Aij->n, Aij->n, S->ku[p], S->kl[p]);

					/* solve Ai * Vi = Bi */
					directSolver_SolveForRHS( handler, Vi->m, Vi->aij, Bi->aij);
					debug("Solving Done");

					Vit = block_ExtractTip( Vi, _TOP_SECTION_, _ROWMAJOR_ );
					Vib = block_ExtractTip( Vi, _BOTTOM_SECTION_, _ROWMAJOR_ );
					Bib2 = block_ExtractTip( Bi, _BOTTOM_SECTION_, _COLMAJOR_ );
					block_Deallocate( Bi);
					block_Deallocate( Vi);
					debug("All Done");
				}

				else{

					Vit = block_CreateEmptyBlock( S->kl[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _TOP_SECTION_ );
					Vib = block_CreateEmptyBlock( S->ku[p], S->ku[p], S->ku[p], S->kl[p], _V_BLOCK_, _BOTTOM_SECTION_ );
					Bib2 = blockingBi(S, BiTmp, Vit, Vib, p, handler);
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
					Cit2 = blockingCi(S, CiTmp, Wit, Wib, p, handler);
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


/**
 * @brief 			Asyncronous: Worker recive the top part of the next section and/or the bottom part
					of the previous section.

					This function solve the backward solution of the system, every node  need to use
				 	Bib and Cit, however the first partiton and the last partition of the spike system
					only need one, Bib for the first partition and Cit for the last partition. When the
					master is working the master take the first partiton.

 * @param S			Spike Schedule.
 * @param Bib		Bottom part of Bi, its possible to get it from workerSolveAndSendTips.
 * @param Cit		Top part of Ci, its possible to get it from workerSolveAndSendTips.
 * @param master	Master node id.
 * @param handler 	Direct solver.
 */
void workerSolveBackward(dm_schedule_t* S, block_t* Bib, block_t* Cit, integer_t master, DirectSolverHander_t *handler){

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

				block_Deallocate ( xb_prev);
				break;
			}
		}
	}
	sendBlockPacked(xi, master, XI_TAG);
	block_Deallocate (xi);
	block_Deallocate (fi);
}


/**
 * @brief 			Master work for the first partition of spike system. (Factoritzation
					part).

					This function does the same than what the workers do in workerSolveAndSendTips
					but with some optimizations because the master have already the data on memory
					and its not needed to send this data. Also this function will insert the tips
					to the reduced system like the gatherReducedSystem does for the rest of the 
					system.

 * @param handler 	Direct solver.
 * @param S			Spike Schedule.
 * @param A			Original Matrix.
 * @param f			Block with the right hand sides.
 * @param R			Reduced system.
 * @param Bib		Bottom part of Bi.
 * @param xr		Solution of the Reduced system.
 * @param nrhs		Number of right hand sides of the spike system.
 */
void masterWorkFactorize(DirectSolverHander_t *handler, dm_schedule_t* S, matrix_t* A, block_t* f, matrix_t* R, block_t* Bib, block_t* xr, integer_t nrhs){

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
	//matrix_Deallocate(Aij);
}


/**
 * @brief 			Master work for the first partition of spike system. (Backward part).

					This function does the same than what the workers do in workerSolveAndSendTips
					but with some optimizations because the master have already the data on memory
					and its not needed to send this data. Also this function will insert the tips
					to the reduced system like the gatherReducedSystem does for the rest of the 
					system.

 * @param S			Spike Schedule.
 * @param yr		Block with the right hand sides.
 * @param f			Block with the right hand sides.
 * @param x			Reduced system.
 * @param Bib		Bottom part of Bi.
 * @param handler 	Direct solver.
 */
void masterWorkBackward(dm_schedule_t* S, block_t* yr, block_t* f, block_t* x, block_t* Bib, DirectSolverHander_t *handler){

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

/**
 * @brief 		Debuging functions for MPI.

 * @param file	File name of the logfile.
 * @param func	Name of the function who call this function.
 * @param type	DEBUG, ERROR, STAT.
 * @param fmt	String format of the output like "printf".
 * @param ...	Other parameters to fit in fmt string.

 */
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
