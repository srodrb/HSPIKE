/*
 * =====================================================================================
 *
 *       Filename:  spike_datatypes.h
 *
 *    Description:  Datatypes definition for SPIKE solver
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:35:01
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#ifndef _SPIKE_DATATYPES_H_
	#define _SPIKE_DATATYPES_H_



	typedef double   complex_t;
	typedef double   real_t;
	typedef int      integer_t;
	typedef int      Error_t;

	extern const integer_t ALIGN_INT;
	extern const integer_t ALIGN_REAL;
	extern const integer_t ALIGN_COMPLEX;

	extern const complex_t __unit;

	extern const Error_t SPIKE_SUCCESS;


#endif /* end of _SPIKE_DATATYPES_H_ definition */
