#include "spike_memory.h"


unsigned int cnt_alloc = 0;
unsigned int cnt_free  = 0;

// TODO provide a mechanism for malloc aligned memory with non-intel compilers

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

	cnt_alloc += 1;

	return (buffer);
};


void spike_free ( void* ptr )
{
	#ifdef __INTEL_COMPILER
		if ( ptr ) { _mm_free( ptr ); }
	#else
		if ( ptr)  { free ( ptr ); }
	#endif

	cnt_free += 1;
};

void spike_nullify ( void* ptr )
{
	spike_free(ptr); ptr = NULL;
};