/*
 * =====================================================================================
 *
 *       Filename:  spike_common.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/07/16 15:29:21
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include "spike_common.h"

FILE* spike_fopen( const char* filename, const char* mode)
{
	FILE* f = fopen( filename, mode );

	if ( f == NULL )
	{
		fprintf(stderr, "Cant open %f correctly!\n", filename);
		abort();
	}

	return (f);
};

void spike_fclose( FILE* f )
{
	fclose( f );
};

void spike_fwrite( void* ptr, size_t size, size_t nmemb, FILE* stream)
{
	size_t n;

	if( (n=fwrite(ptr, size, nmemb, stream)) != n )
	{
		fprintf(stderr, "Cant write %lu elements of size %lu correctly!\n", nmemb, size);
		abort();
	}
};

void spike_fread( void* ptr, size_t size, size_t nmemb, FILE* stream)
{
	size_t n;

	if( (n=fread(ptr, size, nmemb, stream)) != n )
	{
		fprintf(stderr, "Cant read %lu elements of size %lu correctly!\n", nmemb, size);
		abort();
	}
};

