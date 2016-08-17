#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#ifdef NDEBUG
	#define debug(M, ...)
#else		
	#define debug(M, ...) my_debug(__FILE__, __LINE__, __func__, M, __VA_ARGS__)
#endif

void my_debug(char *file, int line, const char *func, const char *fmt, ...){

	FILE *fp;
	fp = fopen ("try.log", "a");
	//fprintf(fp, "DEBUG %s:%d:%s\n",file, line, func);

	va_list args;
	va_start(args, fmt);
	fprintf(fp, "DEBUG %s:%d:%s:", file, line, func);
	vfprintf(fp, fmt, args);
	va_end(args);
	
	fclose (fp);
}
int main(int argc, char *argv[]){
	
	int a=5;
	
	debug("hola %E\n", 0.000000034444426);
	return 0;
}
