/*
 * =====================================================================================
 *
 *       Filename:  spike_datatypes.c
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:34:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "spike_datatypes.h"

#ifdef __GNUC__
  #define __restrict __restrict__
#else
  #define __restrict restrict
#endif

#if defined (_DATATYPE_Z_) || defined (_DATATYPE_C_)
  /* complex arithmetic */
  const complex_t __unit = (complex_t) {1.0, 1.0};
  const complex_t __zero = (complex_t) {0.0, 0.0};

#else
  /* real arithmetic */
  const complex_t __unit = (complex_t) 1.0;
  const complex_t __zero = (complex_t) 0.0;

#endif

const Bool_t True  = 1;
const Bool_t False = 0;
const Error_t SPIKE_SUCCESS = 1;

Bool_t number_IsLessThan( complex_t a, complex_t b )
{
  #if defined (_COMPLEX_ARITHMETIC_)
    real_t amod = (a.real * a.real) + (a.imag * a.imag);
    real_t bmod = (b.real * b.real) + (b.imag * b.imag);

    if ( amod < bmod ) return (True);
  #else
    if ( a < b ) return (True);
  #endif

  return (False);
};

Bool_t number_IsEqual( complex_t a, complex_t b )
{
  #ifdef 	_COMPLEX_ARITHMETIC_
    real_t amod = (a.real * a.real) + (a.imag * a.imag);
    real_t bmod = (b.real * b.real) + (b.imag * b.imag);

    if ( amod == bmod ) return (True);
  #else
    if ( a == b ) return (True);
  #endif

  return (False);
};
