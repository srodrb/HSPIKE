/*
 * =====================================================================================
 *
 *       Filename:  spike_interfaces.h
 *
 *    Description:  High-level interfaces for the heterogeneous spike solver.
 *
 *        Version:  1.0
 *        Created:  26/07/16 07:26:00
 *       Revision:  none
 *       Compiler:  icc / gcc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#ifndef _SPIKE_INTERFACES_
	#define _SPIKE_INTERFACES_

 	#include "spike_analysis.h"

 	typedef struct { float real ; float  imag; } complex8;
 	typedef struct { double real; double imag; } complex16;


 	Error_t    sspike_core_host    (const integer_t n,
									const integer_t nnz,
									const integer_t nrhs,
									integer_t *restrict colind,
									integer_t *restrict rowptr,
									float     *restrict aij,
									float     *restrict xij,
									float     *restrict bij);

 	Error_t    dspike_core_host    (const integer_t n,
									const integer_t nnz,
									const integer_t nrhs,
									integer_t *restrict colind,
									integer_t *restrict rowptr,
									double    *restrict aij,
									double    *restrict xij,
									double    *restrict bij);

 	Error_t    cspike_core_host    (const integer_t n,
									const integer_t nnz,
									const integer_t nrhs,
									integer_t *restrict colind,
									integer_t *restrict rowptr,
									complex8  *restrict aij,
									complex8  *restrict xij,
									complex8  *restrict bij);

 	Error_t    zspike_core_host    (const integer_t n,
									const integer_t nnz,
									const integer_t nrhs,
									integer_t *restrict colind,
									integer_t *restrict rowptr,
									complex16 *restrict aij,
									complex16 *restrict xij,
									complex16 *restrict bij);

 	/* column-blocking routines */
 	Error_t zspike_core_host_blocking (
 		const integer_t n,
		const integer_t nnz,
		const integer_t nrhs,
		integer_t *restrict colind,
		integer_t *restrict rowptr,
		complex16 *restrict aij,
		complex16 *restrict xij,
		complex16 *restrict bij);


#endif /* end of _SPIKE_INTERFACES_ definition */