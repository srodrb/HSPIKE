/*
 * =====================================================================================
 *
 *       Filename:  main.c
 *
 *    Description:  SPIKE usage demonstration
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:32:39
 *       Revision:  none
 *       Compiler:  mpicc
 *
 *         Author:  Albert Coca Abelló
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */
#include "spike_mpi.h"

/* -------------------------------------------------------------------- */
/* .. MPI: Send Aij to Slaves.
/* -------------------------------------------------------------------- */
void sendAij (matrix_t *Aij, integer_t p){
	
	//printf("sending: %d, %d, %d, %d, %d\n", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->K);
	MPI_Send(&Aij->n,     5, 		     MPI_INT, 	      p, 0, MPI_COMM_WORLD);
	MPI_Send(Aij->colind, Aij->nnz, 	     MPI_INT, 	      p, 0, MPI_COMM_WORLD);
	MPI_Send(Aij->rowptr, Aij->n+1, 	     MPI_INT, 	      p, 0, MPI_COMM_WORLD);
	MPI_Send(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_,  p, 0, MPI_COMM_WORLD);
	//printf("Sended\n");
};


/* -------------------------------------------------------------------- */
/* .. MPI: Recv Aij From master
/* -------------------------------------------------------------------- */
matrix_t* recvAij (integer_t master){
	
	int t[5];
	MPI_Status  status;

	MPI_Recv(t, 5, MPI_INT, master, 0, MPI_COMM_WORLD, &status);

	matrix_t* Aij = matrix_CreateEmptyMatrix( t[0], t[1] );
	Aij->ku = t[2];
	Aij->kl = t[3];
	Aij->K  = t[4];

	//printf("Recived: %d, %d, %d, %d, %d\n", Aij->n, Aij->nnz, Aij->ku, Aij->kl, Aij->K);

	MPI_Recv(Aij->colind, Aij->nnz, 	     MPI_INT, 	     master, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(Aij->rowptr, Aij->n+1, 	     MPI_INT, 	     master, 0, MPI_COMM_WORLD, &status);
	MPI_Recv(Aij->aij,    Aij->nnz*_MPI_COUNT_, _MPI_COMPLEX_T_, master, 0, MPI_COMM_WORLD, &status);
	//matrix_Print( Aij, NULL);

	return Aij;
}

void sendBlock (block_t *b, integer_t p){
	MPI_Send(&b, 6,                      MPI_INT, 	     p, 0, MPI_COMM_WORLD);
	MPI_Send(b->aij,  (b->m)*(b->n)*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD);
	
	
}
block_t* recvBlock (integer_t p){

	int t[6];
	MPI_Status  status;

	MPI_Recv(t, 6, 	     MPI_INT, 	     p, 0, MPI_COMM_WORLD, &status);

	block_t *b = block_CreateEmptyBlock(t[4], t[5], t[0], t[1], t[2], t[3]);
	debug ("Recived %d, %d, %d, %d, %d, %d", t[4], t[5], t[0], t[1], t[2], t[3]);
	
	MPI_Recv(b->aij, (b->n)*(b->m)*_MPI_COUNT_, _MPI_COMPLEX_T_, p, 0, MPI_COMM_WORLD, &status);
	return b;
	
}

void my_debug(char *file, integer_t line, const char *func, const char *fmt, ...){

	FILE *fp;
	integer_t rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	char fpName[20];
	sprintf(fpName, "Mpi_Debug%d.log", rank);
	fp = fopen (fpName, "a");
	//fprintf(fp, "DEBUG %s:%d:%s\n",file, line, func);

	va_list args;
	va_start(args, fmt);
	fprintf(fp, "DEBUG %s:%d:%s:", file, line, func);
	vfprintf(fp, fmt, args);
	va_end(args);
	
	fclose (fp);
}
