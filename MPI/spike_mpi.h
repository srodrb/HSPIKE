//#include "spike_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void sendAij (matrix_t *Aij, integer_t p);
matrix_t* recvAij (integer_t master);
void debug_mpi(integer_t rank, char *msg);
