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

	#include "spike_algebra.h"

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
		integer_t p;

		integer_t max_n; // value of max rows
		integer_t max_m; // value of max cols

		integer_t *n;
		integer_t *ku;
		integer_t *kl;

	} sm_schedule_t;

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
	sm_schedule_t* spike_solve_analysis ( matrix_t* A, const integer_t nrhs, const integer_t p );

	void schedule_Destroy( sm_schedule_t* S );

	void schedule_Print(sm_schedule_t* S);

#endif /* end of _SPIKE_ANALYSIS_H_ definition */
