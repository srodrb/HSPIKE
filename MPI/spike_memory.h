/*
 * =====================================================================================
 *
 *       Filename:  spike_memory.h
 *
 *    Description:  Memory interfaces
 *
 *        Version:  1.0
 *        Created:  21/06/16 10:44:17
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  Samuel Rodriguez Bernabeu
 *   Organization:  Barcelona Supercomputing Center
 *
 * =====================================================================================
 */

#ifndef _SPIKE_MEMORY_H_
	#define _SPIKE_MEMORY_H_

	#include <stdio.h>
	#include <stdlib.h>

	extern const int ALIGN_INT;
	extern const int ALIGN_REAL;
	extern const int ALIGN_COMPLEX;

	void* spike_malloc    ( const int alignment, const int nmemb, const size_t size);
	void  spike_free      ( void* ptr );
	void  spike_nullify   ( void* ptr );

#endif /* end of _SPIKE_MEMORY_H_ definition */
