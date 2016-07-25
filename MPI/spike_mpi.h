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

#include "spike_analysis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <mpi.h>

#ifdef NDEBUG
	#define debug(M, ...)
	#define error(M, ...)
#else		
	#define debug(M, ...) my_debug(__FILE__, __LINE__, __func__, "\x1B[34mDEBUG", M, ##__VA_ARGS__)
	#define stat(M, ...)  my_debug(__FILE__, __LINE__, __func__, "STAT", M, ##__VA_ARGS__)
	#define error(M, ...) my_debug(__FILE__, __LINE__, __func__, "\x1B[31mERROR", M, ##__VA_ARGS__)
#endif


/*----------------------------------------------------
-	Send and Recive Matrix Functions
-----------------------------------------------------*/
void sendMatrix		  (matrix_t *Aij, integer_t p);
void IsendMatrix	  (matrix_t *Aij, integer_t p);
void sendMatrixPacked (matrix_t *Aij, integer_t p);

matrix_t* recvMatrix	   (integer_t p);
matrix_t* recvMatrixPacked (integer_t p);

/*----------------------------------------------------
-	Send and Recive Block Functions
-----------------------------------------------------*/
void sendBlock		 (block_t *b, integer_t p);
void sendBlockPacked (block_t *b, integer_t p);

block_t* recvBlock 		 (integer_t p);
block_t* recvBlockPacked (integer_t p);

matrix_t* recvAndAddBlock		(integer_t *ku, integer_t *n, integer_t *kl); //NOT FINISHED
matrix_t* recvAndAddBlockPacked (integer_t *ku, integer_t *n, integer_t *kl, integer_t numBlocks);

/*----------------------------------------------------
-	Debug Functions
-----------------------------------------------------*/
void my_debug(char *file, int line, const char *func, char *type, const char *fmt, ...);



