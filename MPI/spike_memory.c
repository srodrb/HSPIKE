#include "spike_memory.h"

const int ALIGN_INT     = 64;
const int ALIGN_REAL    = 64;
const int ALIGN_COMPLEX = 64;

void* spike_malloc( const int alignment, const int nmemb, const size_t size)
{

#ifdef __INTEL_COMPILER
	void *buffer = _mm_malloc( size * nmemb, alignment );

#else
	void *buffer = malloc( size * nmemb );
#endif

	if ( buffer == NULL )
	{
		fprintf(stderr, "Cant allocate %d elements of size %lu correctly\n", nmemb, size);
		abort();
	}

	return (buffer);
};


void spike_free ( void* ptr )
{
	#ifdef __INTEL_COMPILER
		_mm_free( ptr );
	#else
	free ( ptr );
	#endif
};

void spike_nullify ( void* ptr )
{
	spike_free(ptr); ptr = NULL;
};
