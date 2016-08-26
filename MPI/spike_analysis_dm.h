/*
 * =====================================================================================
 *
 *       Filename:  spike_analysis.h
 *
 *    Description:  Analysis of sparse matrix routines
 *
 *        Version:  1.0
 *        Created:  21/06/16 15:32:42
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#ifndef _SPIKE_ANALYSIS_H_
	#define _SPIKE_ANALYSIS_H_

	#include "spike_matrix.h"
 	#include <math.h>
	#include <sys/sysinfo.h> /* memory structure */
	#include <mpi.h>

	//Configuration
	#define BLOCKING 		1
	#define MASTER_WORKING 	0

	/*
	 * Shared memory version of the schedule structure
	 * n: (array, size p+1) dimensions of the blocks
	 * ku: (array, size p) upper bandwidth of the blocks
	 * kl: (array, size p) lower bandwidth of the blocks
	 *
	 * size(n)=(p+1) because we would like to formulate lopps in a natural fashion.
	 */
	typedef struct
	{
		integer_t p; /* number of partitions in which the matrix is divided */

		integer_t max_n; // value of max rows 
		integer_t max_m; // value of max colssy

		integer_t max_nrhs;
		integer_t blockingDistance;

		integer_t *n;    /* index of last row of each block for the original system */
		integer_t *r;    /* index of last row of each block for the reduced system */
		integer_t *ku;   /* upper bandwidth of the block */
		integer_t *kl;   /* lower bandwidth of the block */

	} dm_schedule_t;

	/*
	 * The solve strategy could be fairly complex.
	 *
	 * For shared memory systems with multiple accelerators
	 * we rely on an analytic model based on the cost of the
	 * factorizations and triangular sweeps to be performed.
	 *
	 * On top of that, we'd like either to reduce the memory
	 * consumption or solve the system faster.
	 */
	dm_schedule_t* spike_solve_analysis ( matrix_t* A, const integer_t nrhs );

	void schedule_Destroy( dm_schedule_t* S );

	void schedule_Print(dm_schedule_t* S);

	/*-------------------------------------------------------------------*/
	uLong_t get_maximum_av_host_memory( void );

	integer_t compute_optimal_number_of_partitions( matrix_t *A, integer_t nrhs, uLong_t HostMem );


#endif /* end of _SPIKE_ANALYSIS_H_ definition */
