/*
 * =====================================================================================
 *
 *       Filename:  spike_common.h
 *
 *    Description:  Common routines for spike
 *
 *        Version:  1.0
 *        Created:  04/07/16 15:26:50
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef _SPIKE_COMMON_H_
	#define _SPIKE_COMMON_H_

	#include <stdio.h>
	#include <stdlib.h>


	FILE* spike_fopen( const char* filename, const char* mode);

	void spike_fclose( FILE* f );

	void spike_fwrite( void* ptr, size_t size, size_t nmemb, FILE* stream);

	void spike_fread( void* ptr, size_t size, size_t nmemb, FILE* stream);

#endif /* end of  _SPIKE_COMMON_H_ definition */
