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

timer_t GetReferenceTime(void)
{
	timer_t tseconds = 0.0;
	struct timeval mytime;
	gettimeofday( &mytime, (struct timezone*) 0);
	tseconds = (timer_t) (mytime.tv_sec + mytime.tv_usec * 1.0e-6);
	return (tseconds);
};

void CheckPreprocessorMacros (void)
{
	char *datatype;
	char *mpi_support;
	char *complex_support;

	#if (_MPI_SUPPORT_)
		mpi_support = "mpi support enabled";
	#else
		mpi_support = "mpi support disabled";
	#endif

	#if defined (_DATATYPE_Z_) // double complex
		#define _COMPLEX_ARITHMETIC_
		datatype = "Double precision complex";
	#elif defined (_DATATYPE_C_) // complex float
		#define _COMPLEX_ARITHMETIC_
		datatype = "Single precision complex";
	#elif defined (_DATATYPE_D_) // double precision float
		datatype = "Double precision";
	#else // single precision float
		datatype = "Single precision";
	#endif

	#if defined (_COMPLEX_ARITHMETIC_)
		complex_support = "Complex arithmetic support enabled";
	#else
		complex_support = "Complex arithmetic support disabled";
	#endif

	fprintf(stderr, "\n%s:\n\t"
		"Data type           :        %s\n\t"
		"MPI support         :        %s\n\t"
		"Complex arithmetic  :        %s\n\n", 
		__FUNCTION__, datatype, mpi_support, complex_support );
};
