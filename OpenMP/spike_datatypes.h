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

	#ifdef _MPI_SUPPORT_
		#include <mpi.h>
	#endif


#if defined (_DATATYPE_Z_) // double complex
	#define _COMPLEX_ARITHMETIC_
	typedef double   real_t;

	typedef struct {
		real_t real;
		real_t imag;
	} complex_number;

	typedef complex_number   complex_t;

	#if defined (_MPI_SUPPORT_)
		#define _MPI_COMPLEX_T_  MPI_DOUBLE * 2
		#define _MPI_REAL_T_  MPI_DOUBLE
	#endif

#elif defined (_MPI_SUPPORT_) // complex float
	#define _COMPLEX_ARITHMETIC_

	typedef float   real_t;

	typedef struct {
		real_t real;
		real_t imag;
	} complex_number;

	typedef complex_number   complex_t;

	#if defined (_MPI_SUPPORT_)
		#define _MPI_COMPLEX_T_  MPI_FLOAT * 2
		#define _MPI_REAL_T_  MPI_FLOAT
	#endif

#elif defined (_DATATYPE_D_) // double precision float
	typedef double   real_t;
	typedef double   complex_t;

	#if defined (_MPI_SUPPORT_)
		#define _MPI_COMPLEX_T_  MPI_DOUBLE * 2
		#define _MPI_REAL_T_  MPI_DOUBLE
	#endif

#else // single precision float
	typedef float   real_t;
	typedef float   complex_t;

	#if defined (_MPI_SUPPORT_)
		#define _MPI_COMPLEX_T_  MPI_FLOAT
		#define _MPI_REAL_T_  MPI_FLOAT
	#endif

#endif

	typedef int      integer_t;
	typedef int      Error_t;
	typedef int      Bool_t;

	#if defined (_MPI_SUPPORT_)
		#define _MPI_INTEGER_T_  MPI_INT
	#endif

	extern const complex_t __unit;
	extern const complex_t __zero;

	extern const Bool_t True;
	extern const Bool_t False;

	extern const Error_t SPIKE_SUCCESS;

	Bool_t number_IsLessThan( complex_t a, complex_t b );
	Bool_t number_IsEqual( complex_t a, complex_t b );

#endif /* end of _SPIKE_DATATYPES_H_ definition */
