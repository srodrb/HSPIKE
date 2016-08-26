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
#include <mpi.h>
void blockingFi(dm_schedule_t* S, block_t* fi, block_t* yit, block_t* yib, integer_t nrhs, integer_t p, DirectSolverHander_t *handler);
block_t* blockingBi(dm_schedule_t* S, matrix_t* BiTmp, block_t* Vit, block_t* Vib, integer_t p, DirectSolverHander_t *handler);
block_t* blockingCi(dm_schedule_t* S, matrix_t* CiTmp, block_t* Wit, block_t* Wib, integer_t p, DirectSolverHander_t *handler);
