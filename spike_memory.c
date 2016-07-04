#include "spike_memory.h"

void* spike_malloc(const int alignment, const int nmemb, const size_t size)
{
	void *buffer = malloc( size * nmemb);

	if ( buffer == NULL )
	{
		fprintf(stderr, "Cant allocate %d elements of size %lu correctly\n", nmemb, size);
		abort();
	}

	return (buffer);
};

void spike_nullify ( void* ptr )
{
	free(ptr); ptr = NULL;
};

void spike_free ( void* ptr )
{
	free(ptr);
};



