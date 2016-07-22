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

/* -------------------------------------------------------------------- */
/* .. Function to send matrix_t to process p, p must recive 
/* .. this matrix with recvMatrix function.
/* -------------------------------------------------------------------- */
void sendMatrix (matrix_t *Aij, integer_t p){
	
	MPI_CheckCall(MPI_Send(&Aij->n,     5,					   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->colind, Aij->nnz,			   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->rowptr, Aij->n+1,			   MPI_INT, 	   p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD));
};

/* -------------------------------------------------------------------- */
/* .. Function to send matrix_t to process p, p must recive 
/* .. this matrix with recvMatrix function.
/* -------------------------------------------------------------------- */
void IsendMatrix (matrix_t *Aij, integer_t p){
	
	MPI_Request request;
	MPI_CheckCall(MPI_Isend(&Aij->n,     5,			    		MPI_INT, 	    p, 0, MPI_COMM_WORLD,&request));
	MPI_CheckCall(MPI_Isend(Aij->colind, Aij->nnz,		    	MPI_INT, 	    p, 0, MPI_COMM_WORLD,&request));
	MPI_CheckCall(MPI_Isend(Aij->rowptr, Aij->n+1,				MPI_INT, 	    p, 0, MPI_COMM_WORLD,&request));
	MPI_CheckCall(MPI_Isend(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD,&request));

};

/* -------------------------------------------------------------------- */
/* .. Function to send packed matrix_t to process p, p must recive 
/* .. this matrix with recvMatrix function.
/* -------------------------------------------------------------------- */
void sendMatrixPacked (matrix_t *Aij, integer_t p){

	integer_t position = 0, i;
	integer_t buffSize = (Aij->n+1 + Aij->nnz + 5)*sizeof(integer_t) + Aij->nnz*sizeof(complex_t);
	char *buff = (char*) malloc(buffSize*sizeof(char));
	
	MPI_Pack(Aij		, 5		  			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->colind, Aij->nnz			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->rowptr, Aij->n+1			  , MPI_INT		   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(Aij->aij	, Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_ , buff, buffSize, &position, MPI_COMM_WORLD);
	debug("Values: %d, %d, %d, %d, %d, buffSize: %d", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->K, buffSize);

	MPI_Send(buff, position, MPI_PACKED, p, 0, MPI_COMM_WORLD);
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
	Aij->ku = t[2];
	Aij->kl = t[3];
	Aij->K  = t[4];

	MPI_CheckCall(MPI_Recv(Aij->colind, Aij->nnz, 	           MPI_INT,		   p, 0, MPI_COMM_WORLD, &status));
	MPI_CheckCall(MPI_Recv(Aij->rowptr, Aij->n+1, 	     	   MPI_INT, 	   p, 0, MPI_COMM_WORLD, &status));
	MPI_CheckCall(MPI_Recv(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD, &status));

	return Aij;
}

/* -------------------------------------------------------------------- */
/* .. Function for recive matrix_t from process p, p must send 
/* .. this matrix with sendMatrix function.
/* -------------------------------------------------------------------- */
matrix_t* recvMatrixPacked (integer_t p){
	
	integer_t buffSize = 0, position = 0, i;
	MPI_Status  status;
	integer_t t[5];

	MPI_Probe(p, 0, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_PACKED, &buffSize);
	char* buff = (char*)malloc(sizeof(char) * buffSize);

	MPI_Recv(buff, buffSize, MPI_PACKED, p, 0, MPI_COMM_WORLD, &status);
	MPI_Unpack(buff, buffSize, &position, t, 5, MPI_INT, MPI_COMM_WORLD);
	debug("Values: %d, %d, %d, %d, %d, buffSize: %d", t[0], t[1], t[2], t[3], t[4], buffSize);

	matrix_t* Aij = matrix_CreateEmptyMatrix( t[0], t[1] );
	Aij->ku = t[2];
	Aij->kl = t[3];
	Aij->K  = t[4];
	
	MPI_Unpack(buff, buffSize, &position, Aij->colind, Aij->nnz			   , MPI_INT	   , MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, Aij->rowptr, Aij->n+1			   , MPI_INT	   , MPI_COMM_WORLD);
	MPI_Unpack(buff, buffSize, &position, Aij->aij	 , Aij->nnz*_MPI_COUNT_,_MPI_COMPLEX_T_, MPI_COMM_WORLD);

	return Aij;
}

/* -------------------------------------------------------------------- */
/* .. Function to send block_t to process p, p must recive 
/* .. this matrix with recvBlock function.
/* -------------------------------------------------------------------- */
void sendBlock (block_t *b, integer_t p){
	
	integer_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;

	MPI_CheckCall(MPI_Send(b, 6, MPI_INT, p, 0, MPI_COMM_WORLD));
	MPI_CheckCall(MPI_Send(b->aij,  sendCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD));
	block_Print(b, "Sent Block");

}

/* -------------------------------------------------------------------- */
/* .. Function to send block_t to process p, p must recive 
/* .. this matrix with recvBlock function.
/* -------------------------------------------------------------------- */
void sendBlockPacked (block_t *b, integer_t p){
	
	integer_t sendCount = (b->n)*(b->m)*_MPI_COUNT_;
	integer_t buffSize = 6*sizeof(integer_t) + sendCount*sizeof(complex_t);
	char *buff = (char*) malloc(buffSize*sizeof(char));
	
	integer_t position = 0;
	debug("Sending Block Info: n:%d, m:%d, ku:%d, kl:%d, type:%d, section:%d", b->n, b->m, b->ku, b->kl, b->type, b->section);
	MPI_Pack(b	   , 6		  ,  MPI_INT	   , buff, buffSize, &position, MPI_COMM_WORLD);
	MPI_Pack(b->aij, sendCount, _MPI_COMPLEX_T_, buff, buffSize, &position, MPI_COMM_WORLD);

	MPI_Send(buff, position, MPI_PACKED, p, 0, MPI_COMM_WORLD);
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
	debug("Recived Info: n:%d, m:%d, ku:%d, kl:%d, type:%d, section:%d", b->n, b->m, b->ku, b->kl, b->type, b->section);
	recvCount = (b->n)*(b->m)*_MPI_COUNT_;
	
	MPI_CheckCall(MPI_Recv(b->aij, recvCount, _MPI_COMPLEX_T_, p, 1, MPI_COMM_WORLD, &status));

	return b;
}

/* -------------------------------------------------------------------- */
/* .. Function for recive block_t from process p, p must send 
/* .. this block with sendBlock function.
/* -------------------------------------------------------------------- */
block_t* recvBlockPacked (integer_t p){

	integer_t buffSize=0, position = 0;
	MPI_Status  status;
	integer_t t[6];

	MPI_Probe(p, 0, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_PACKED, &buffSize);
	char* buff = (char*)malloc(sizeof(char) * buffSize);
	//char buff[500];
	debug("BuffSize %d From %d", buffSize, status.MPI_SOURCE);

	MPI_Recv(buff, buffSize, MPI_PACKED, p, 0, MPI_COMM_WORLD, &status);

	MPI_Unpack(buff, buffSize, &position, t, 6, MPI_INT, MPI_COMM_WORLD);
	debug("Recived Block Info: n:%d, m:%d, ku:%d, kl:%d, type:%d, section:%d", t[2], t[3], t[4], t[5], t[0], t[1]);
	block_t *b = block_CreateEmptyBlock(t[2], t[3], t[4], t[5], t[0], t[1]);
	integer_t recvCount = (b->n)*(b->m)*_MPI_COUNT_;
	MPI_Unpack(buff, buffSize, &position, b->aij, recvCount , _MPI_COMPLEX_T_ , MPI_COMM_WORLD);
	return b;
}

/* -------------------------------------------------------------------- */
/* .. Function for recive block_t from all process in Asyncronous way,
/* .. The sending process must send the block with sendBlockPacked function.
/* -------------------------------------------------------------------- */
matrix_t* recvAndAddBlockPacked(integer_t *ku, integer_t *n, integer_t *kl, integer_t numBlocks){
	
	integer_t rank, size, index, i,j;	
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Status  status;
	integer_t t[6];
	integer_t position = 0, buffSize = 0;
	matrix_t* R = matrix_CreateEmptyReducedSystem( size-1, n, ku, kl);

	for(i=0; i<numBlocks; i++){

		position = 0;

		MPI_CheckCall(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
		MPI_CheckCall(MPI_Get_count(&status, MPI_PACKED, &buffSize));
		char* buff = (char*)malloc(sizeof(char) * buffSize);

		MPI_CheckCall(MPI_Recv(buff, buffSize, MPI_PACKED, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status));

		MPI_CheckCall(MPI_Unpack(buff, buffSize, &position, t, 6, MPI_INT, MPI_COMM_WORLD));
		block_t *b = block_CreateEmptyBlock(t[2], t[3], t[4], t[5], t[0], t[1]);
		integer_t recvCount = (b->n)*(b->m)*_MPI_COUNT_;

		MPI_CheckCall(MPI_Unpack(buff, buffSize, &position, b->aij, recvCount , _MPI_COMPLEX_T_ , MPI_COMM_WORLD));
		matrix_AddBlockToReducedSystem(size-1, status.MPI_SOURCE-1, n, ku, kl, R, b);

		free( buff );
	}

	
	return R;
}

/* -------------------------------------------------------------------- */
/* NOT FINISHED: Classical implementation of asyncronous recive
/* block_t. NOT FINISHED.
/* -------------------------------------------------------------------- */
matrix_t* recvAndAddBlock(integer_t *ku, integer_t *n, integer_t *kl){

	integer_t rank, size, index, i,j;	
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	MPI_Status  status;
	
	integer_t buffSize = 4*(size-2);
	MPI_Request *request = (MPI_Request *) spike_malloc( ALIGN_INT, buffSize, sizeof(MPI_Request));
	block_t **recvBlocks = (block_t **) spike_malloc( ALIGN_INT, buffSize, sizeof(block_t*));

	integer_t **recvInfo = (integer_t **) spike_malloc( ALIGN_INT, buffSize, sizeof(integer_t*));
	for(i = 0; i < buffSize; i++) recvInfo[i] = (integer_t *) spike_malloc( ALIGN_INT, 6, sizeof(integer_t));

	matrix_t* R = matrix_CreateEmptyReducedSystem( size-1, n, ku, kl );
	
	for(i=2; i<buffSize+2; i++){
		MPI_CheckCall(MPI_Irecv(&recvInfo[i-2][0], 6, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request[i-2]));
		//debug("i: %d, recvInfo[%d], p:%d", i, i-2, i/4+1);
	}
	debug("------------------------------------------");
	int k, recvCount;
	for(i=0; i<buffSize; i++){
		MPI_Waitany(buffSize, request, &index, &status);
		//debug("Reciving Asyncronous Info from %d, index: %d, iter: %d", status.MPI_SOURCE, index, i);
		debug("Recived Info: n:%d, m:%d, ku:%d, kl:%d, type:%d, section:%d", recvInfo[index][2], recvInfo[index][3], recvInfo[index][4], recvInfo[index][5], recvInfo[index][0], recvInfo[index][1]);
		recvBlocks[index] = block_CreateEmptyBlock(recvInfo[index][2], recvInfo[index][3], recvInfo[index][4], recvInfo[index][5], recvInfo[index][0], recvInfo[index][1]);
		//debug("Recived Block Info: n:%d, m:%d, ku:%d, kl:%d", recvBlocks[index]->n, recvBlocks[index]->m, recvBlocks[index]->ku, recvBlocks[index]->kl);
	}

	for(i=2; i<buffSize+2; i++){
		recvCount = recvBlocks[i-2]->n * recvBlocks[i-2]->m * _MPI_COUNT_;
		MPI_CheckCall(MPI_Irecv(recvBlocks[i-2]->aij, recvCount, _MPI_COMPLEX_T_, i/4+1, 1, MPI_COMM_WORLD, &request[i-2]));
	}
	for(i=0; i<buffSize; i++){
		MPI_Waitany(buffSize, request, &index, &status);
		block_Print(recvBlocks[index], "Recived Block");
		//recvCount = recvBlocks[index]->n * recvBlocks[index]->m * _MPI_COUNT_;
		//debug("Reciving Asyncronous Data from %d", status.MPI_SOURCE);
		matrix_AddBlockToReducedSystem(size-1, status.MPI_SOURCE-1, n, ku, kl, R, recvBlocks[index]);
		debug("%d, %d, %d",size-1, status.MPI_SOURCE-1, index);
		//for(k=0; k<recvCount; k++)printf("->%d, ", recvBlocks[index]->aij[k]);
	}
	return R;
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
	
	fclose (fp);
}
