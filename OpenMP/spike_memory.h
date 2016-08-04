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
 	#include <string.h>

 	extern unsigned int cnt_alloc;
 	extern unsigned int cnt_free;

	#define ALIGN_INT     (const int) 16
	#define ALIGN_REAL    (const int) 32
	#define ALIGN_COMPLEX (const int) 32

 	#define spike_malloc(alignemt, nmemb, size) _spike_malloc(alignemt, nmemb, size, __FUNCTION__, __LINE__ )

	void* _spike_malloc   ( const int alignment, const size_t nmemb, const size_t size, const char* function, const int line);
	void  spike_free      ( void* ptr );
	void  spike_nullify   ( void* ptr );

#endif /* end of _SPIKE_MEMORY_H_ definition */
