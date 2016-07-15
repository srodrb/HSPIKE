//#include "spike_matrix.h"
#include "spike_analysis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <mpi.h>

#ifdef NDEBUG
	#define debug(M, ...)
#else		
	#define debug(M, ...) my_debug(__FILE__, __LINE__, __func__, M, ##__VA_ARGS__)
#endif

void sendAij (matrix_t *Aij, integer_t p);
matrix_t* recvAij (integer_t master);
void debug_mpi(char *file, int line, const char *func, const char *fmt, int rank, ...);
void my_debug(char *file, int line, const char *func, const char *fmt, ...);

void sendBlock (block_t *b, integer_t p);
block_t* recvBlock (integer_t p);
