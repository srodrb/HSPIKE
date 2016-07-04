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

		
	typedef float complex_t;
	typedef float real_t;
	typedef int   integer_t;
	
	extern const int datatype;	
	extern const int ALIGN_INT;
	extern const int ALIGN_REAL;
	extern const int ALIGN_COMPLEX;



#endif /* end of _SPIKE_DATATYPES_H_ definition */