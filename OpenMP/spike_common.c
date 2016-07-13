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
		fprintf(stderr, "Cant open %s correctly!\n", filename);
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

	if ( stream == NULL )
	{
		fprintf(stderr, "Invalid stream\n");
		abort();
	}


	if( (n=fwrite(ptr, size, nmemb, stream)) != nmemb )
	{
		fprintf(stderr, "Cant write %lu elements of size %lu correctly!\n", nmemb, size);
		abort();
	}
};

void spike_fread( void* ptr, size_t size, size_t nmemb, FILE* stream)
{
	size_t n;

	if ( stream == NULL )
	{
		fprintf(stderr, "Invalid stream\n");
		abort();
	}


	if( (n=fread(ptr, size, nmemb, stream)) != nmemb )
	{
		fprintf(stderr, "Cant read %lu elements of size %lu correctly!\n", nmemb, size);
		abort();
	}
};

double GetReferenceTime(void)
{
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday( &mytime, (struct timezone*) 0);
	tseconds = (double) (mytime.tv_sec + mytime.tv_usec * 1.0e-6);
	return (tseconds);
};