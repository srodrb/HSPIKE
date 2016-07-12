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


const complex_t __unit = (complex_t) 1.0;
const complex_t __zero = (complex_t) 0.0;

const Bool_t True  = 1;
const Bool_t False = 0;

Bool_t isLessThan( const complex_t a, const complex_t b )
{
  if ( a < b )
    return (True);
  else
    return (False);
};

const Error_t SPIKE_SUCCESS = 1;
