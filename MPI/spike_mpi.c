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

#include "spike_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <mpi.h>

#ifdef NDEBUG
	#define debug(M, ...)
#else		
	#define debug(M,rank, ...) my_debug(__FILE__, __LINE__, __func__, M, rank, __VA_ARGS__)
#endif

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

	matrix_t* Aij = matrix_CreateEmpty( t[0], t[1] );
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

void debug_mpi(char *file, int line, const char *func, const char *fmt, int rank, ...){

	FILE *fp;
	char fpName[50];
	strcpy(fpName, "Slave");
	strcat(fpName, (char)rank+45);
	strcat(fpName, ".info");	
	
	fp = fopen (fpName, "a");

	va_list args;
	va_start(args, fmt);
	fprintf(fp, "DEBUG Slave %d in %s:%d:%s:", rank, file, line, func);
	vfprintf(fp, fmt, args);
	va_end(args);

	fclose (fp);
}
