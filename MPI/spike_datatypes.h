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

	#define _DATATYPE_S_ 0
	#define _DATATYPE_D_ 1
	#define _DATATYPE_C_ 2
	#define _DATATYPE_Z_ 3

	#ifdef MPI_VERSION
		#include <mpi.h>
	#endif


#if DATATYPE == _DATATYPE_Z_ // double complex
	#define _COMPLEX_ARITHMETIC_
	typedef double   real_t;

	typedef struct {
		real_t real;
		real_t imag;
	} complex_number;

	typedef complex_number   complex_t;

	#ifdef MPI_VERSION
		#define _MPI_COMPLEX_T_  MPI_DOUBLE
		#define _MPI_COUNT_ 2
	#endif

#elif DATATYPE == _DATATYPE_C_ // complex float
	#define _COMPLEX_ARITHMETIC_

	typedef float   real_t;

	typedef struct {
		real_t real;
		real_t imag;
	} complex_number;

	typedef complex_number   complex_t;

	#ifdef MPI_VERSION
		#define _MPI_COMPLEX_T_  MPI_FLOAT
		#define _MPI_COUNT_ 2
	#endif

#elif DATATYPE == _DATATYPE_D_ // double precision float
	typedef double   real_t;
	typedef double   complex_t;

	#ifdef MPI_VERSION
		#define _MPI_COMPLEX_T_  MPI_DOUBLE
		#define _MPI_COUNT_ 1
	#endif

#else // single precision float
	typedef float   real_t;
	typedef float   complex_t;

	#ifdef MPI_VERSION
		#define _MPI_COMPLEX_T_  MPI_FLOAT
		#define _MPI_COUNT_ 1
	#endif

#endif

	typedef int      integer_t;
	typedef int      Error_t;
	typedef int      Bool_t;

	extern const complex_t __unit;
	extern const complex_t __zero;

	extern const Bool_t True;
	extern const Bool_t False;

	extern const Error_t SPIKE_SUCCESS;

	Bool_t number_IsLessThan( complex_t a, complex_t b );
	Bool_t number_IsEqual( complex_t a, complex_t b );

#endif /* end of _SPIKE_DATATYPES_H_ definition */
