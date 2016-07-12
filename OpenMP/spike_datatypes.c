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

#if DATATYPE == _DATATYPE_Z_ // double complex
  const complex_t __unit = (complex_t) {1.0, 1.0};
  const complex_t __zero = (complex_t) {0.0, 0.0};

#elif DATATYPE == _DATATYPE_C_ // complex float
  const complex_t __unit = (complex_t) {1.0, 1.0};
  const complex_t __zero = (complex_t) {0.0, 0.0};

#elif DATATYPE == _DATATYPE_D_ // double precision float
  const complex_t __unit = (complex_t) 1.0;
  const complex_t __zero = (complex_t) 0.0;

#else // single precision float
  const complex_t __unit = (complex_t) 1.0;
  const complex_t __zero = (complex_t) 0.0;

#endif

const Bool_t True  = 1;
const Bool_t False = 0;

Bool_t number_IsLessThan( complex_t a, complex_t b )
{
  #ifdef 	_COMPLEX_ARITHMETIC_
    if ( a.real > b.real && a.imag > b.imag) return (True);
  #else
    if ( a < b ) return (True);
  #endif

  return (False);
};

Bool_t number_IsEqual( complex_t a, complex_t b )
{
  #ifdef 	_COMPLEX_ARITHMETIC_
    if ( a.real == b.real && a.imag == b.imag) return (True);
  #else
    if ( a == b ) return (True);
  #endif

  return (False);
}

const Error_t SPIKE_SUCCESS = 1;
