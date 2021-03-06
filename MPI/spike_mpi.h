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

#include "spike_analysis_dm.h"
#include "spike_blocking.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <mpi.h>

#if defined (_NDEBUG)
	#define debug(M, ...)
	#define stat(M, ...)
#else		
	#define debug(M, ...) my_debug(__FILE__, __LINE__, __func__, "\x1B[34mDEBUG", M, ##__VA_ARGS__)
	#define stat(M, ...)  my_debug(__FILE__, __LINE__, __func__, "STAT", M, ##__VA_ARGS__)
#endif

#define error(M, ...) my_debug(__FILE__, __LINE__, __func__, "\x1B[31mERROR", M, ##__VA_ARGS__)

#if defined (_DATATYPE_Z_) // double complex
		#define _MPI_COMPLEX_T_  MPI_DOUBLE
		#define _MPI_COUNT_  2

#elif defined (_DATATYPE_C_) // complex float
		#define _MPI_COMPLEX_T_  MPI_FLOAT
		#define _MPI_COUNT_ 2

#elif defined (_DATATYPE_D_) // double precision float
		#define _MPI_COMPLEX_T_  MPI_DOUBLE
		#define _MPI_COUNT_ 1

#else // single precision float
		#define _MPI_COMPLEX_T_  MPI_FLOAT
		#define _MPI_COUNT_ 1
#endif

#define _MPI_ULONG_ MPI_LONG_LONG

typedef int      integer_t;
typedef int      Error_t;
typedef int      Bool_t;
typedef double   timer_t;

#define I "%d"

#if defined (_MPI_SUPPORT_)
	#define _MPI_INTEGER_T_  MPI_INT
#endif


//Tags for Asyncronous Send/Recv

#define S_TAG		100
#define AIJ_TAG		105
#define FI_TAG		110
#define	BI_TAG		115
#define CI_TAG		120
#define XT_NEXT_TAG 125
#define XT_PREV_TAG 130

#define VIWI_TAG	200
#define YI_TAG		205
#define XI_TAG		210

#define BIG_MSG_SIZE 	1000

/*----------------------------------------------------
-	Send and Recive Matrix Functions
-----------------------------------------------------*/
void sendMatrix		  (matrix_t *Aij, integer_t p);
void IsendMatrix	  (matrix_t *Aij, integer_t p);
void sendMatrixPacked (matrix_t *Aij, integer_t p, integer_t tag);

matrix_t* recvMatrix	   (integer_t p);
matrix_t* recvMatrixPacked (integer_t p, integer_t tag);

/*----------------------------------------------------
-	Send and Recive Block Functions
-----------------------------------------------------*/
void sendBlock		 (block_t *b, integer_t p);
void IsendBlock 	 (block_t *b, integer_t p);
void sendBlockPacked (block_t *b, integer_t p, integer_t tag);

block_t* recvBlock 		 (integer_t p);
block_t* recvBlockPacked (integer_t p, integer_t tag);

matrix_t* recvAndAddBlock		(integer_t *ku, integer_t *n, integer_t *kl); //NOT FINISHED
matrix_t* recvAndAddBlockPacked (integer_t *ku, integer_t *n, integer_t *kl, integer_t numBlocks);

/*----------------------------------------------------
-	Complex Functions
-----------------------------------------------------*/
void scatterSchedule	(dm_schedule_t* S);
void scatterAijBiCiFi	(dm_schedule_t* S, matrix_t* A, block_t* f);
void gatherReducedSystem(dm_schedule_t* S, matrix_t* R, block_t* xr);
void scatterXiFi		(dm_schedule_t* S, block_t* x, block_t* f, block_t* yr);
void gatherXi			(dm_schedule_t* S, block_t* x);

void workerSolveAndSendTips(dm_schedule_t* S, integer_t master, integer_t nrhs, matrix_t* Aij, block_t **Bib, block_t **Cit, DirectSolverHander_t *handler);
void workerSolveAndSendTipsV2(dm_schedule_t* S, integer_t master, integer_t nrhs, matrix_t* Aij, block_t** Vi, block_t** Wi, DirectSolverHander_t *handler);
void workerSolveBackward(dm_schedule_t* S, block_t* Bib, block_t* Cit, integer_t master, DirectSolverHander_t *handler);
void workerSolveBackwardV2(dm_schedule_t* S, block_t* Vi, block_t* Wi, integer_t master, DirectSolverHander_t *handler);

void masterWorkFactorize(DirectSolverHander_t *handler, dm_schedule_t* S, matrix_t* A, block_t* f, matrix_t* R, block_t* Bib, block_t* xr, integer_t nrhs);
void masterWorkBackward(dm_schedule_t* S, block_t* yr, block_t* f, block_t* x, block_t* Bib, DirectSolverHander_t *handler);

/*----------------------------------------------------
-	Debug Functions
-----------------------------------------------------*/
void my_debug(char *file, int line, const char *func, char *type, const char *fmt, ...);

dm_schedule_t* recvSchedulePacked(integer_t p);
void sendSchedulePacked(dm_schedule_t* S, integer_t p);

