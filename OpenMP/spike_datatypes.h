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
 *       Compiler:  icc / gcc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#ifndef _SPIKE_DATATYPES_H_
	#define _SPIKE_DATATYPES_H_

	#undef _COMPLEX_ARITHMETIC_

 	#include <stdlib.h>



#if defined (_DATATYPE_Z_) // double complex
	#define _COMPLEX_ARITHMETIC_

	typedef double   real_t;

	typedef struct {
		double real;
		double imag;
	} complex_number;

	typedef complex_number   complex_t;

	#define F "%.3lf %.3lf"
	/* ensure compatibility with Intel's MKL library */
	/* https://software.intel.com/en-us/node/528405  */
	#define MKL_Complex16 complex_t

	#define _F_ 		"%.3lf"
	#define _PPREF_		z
 
#elif defined (_DATATYPE_C_) // complex float
	#define _COMPLEX_ARITHMETIC_

	typedef float   real_t;

	typedef struct {
		float real;
		float imag;
	} complex_number;

	typedef complex_number   complex_t;

	#define F "%.3f %.3f"
	/* ensure compatibility with Intel's MKL library */
	/* https://software.intel.com/en-us/node/528405  */
	#define MKL_Complex8 complex_t

	#define _F_			"%.3f"
	#define _PPREF_		c

#elif defined (_DATATYPE_D_) // double precision float
	typedef double   real_t;
	typedef double   complex_t;

	#define _F_ "%.3lf"
	#define _PPREF_		d

#else // single precision float
	typedef float   real_t;
	typedef float   complex_t;

	#define _F_ 		"%.3f"
	#define _PPREF_		s


#endif

	typedef int      integer_t;
	typedef int      Error_t;
	typedef int      Bool_t;
	typedef double   spike_timer_t;

	typedef size_t   uLong_t;

	#define _I_ "%d"

	#if defined (_MPI_SUPPORT_)
		#define _MPI_INTEGER_T_  MPI_INT
	#endif

	extern const complex_t __nunit; /* negative unit */
	extern const complex_t __punit; /* positive unit */
	extern const complex_t __zero;  

	extern const Bool_t True;
	extern const Bool_t False;

	extern const Error_t SPIKE_SUCCESS;
	extern const Error_t SPIKE_ERROR;

	Bool_t number_IsLessThan( complex_t a, complex_t b );
	Bool_t number_IsEqual( complex_t a, complex_t b );

#endif /* end of _SPIKE_DATATYPES_H_ definition */
