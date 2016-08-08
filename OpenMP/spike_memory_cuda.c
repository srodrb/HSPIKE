#include "spike_memory.h"


unsigned int cnt_alloc = 0;
unsigned int cnt_free  = 0;

// TODO provide a mechanism for malloc aligned memory with non-intel compilers
void* _spike_malloc   ( const int alignment,
						const size_t nmemb,
						const size_t size,
						const char* function,
						const int line)
{
	if ( nmemb < 0 || size < 0 ){
		fprintf(stderr, "\n%s-%s from %d: Error: nmemb and size must be positive numbers (consider buffer overflow)\n",
			__FUNCTION__, function, line );
		abort();
	}

	void* buffer;

	cudaMallocHost((void**) &buffer, nmemb * size );

	if ( buffer == NULL )
	{
		fprintf(stderr, "\n%s-%s from %d: Cant allocate %zu elements of size %zu correctly\n",
			__FUNCTION__, function, line ,nmemb, size );
		abort();
	}

	cnt_alloc += 1;

	return (buffer);
};


void spike_free ( void* ptr )
{
	cudaFreeHost( ptr );

	cnt_free += 1;
};

void spike_nullify ( void* ptr )
{
	spike_free(ptr); ptr = NULL;
};