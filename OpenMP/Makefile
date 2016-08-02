# options are <INTEL,GNU>
COMPILER =INTEL

# options are: <SUPERLU, CUSOLVER, PARDISO>
BACKEND  =CUSOLVER 

# options are <YES,NO>
DEBUG    =NO

# Data type definition
DEFINES =-D_DATATYPE_D_

OBJS  = spike_datatypes.o spike_memory.o spike_matrix.o \
		spike_algebra.o spike_analysis.o spike_common.o \
		spike_interfaces.o


ifeq ($(COMPILER),INTEL)
	CC      =icc
	CFLAGS  =-qopenmp -std=c99 -O3 -g -restrict
	
	# multi-threaded version
	CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
	CINCS   =-I${MKLROOT}/include
	
	# sequential version
	# CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
	# CINCS   =-I${MKLROOT}/include
else
	CC      =gcc
	CFLAGS  =-std=c99 -g -O2
	CLIBS   =
	CINCS   =
endif

ifeq ($(BACKEND),SUPERLU)
	#CLIBS   +=-L/home/srodrigu/Software/SuperLU_MT_mkl/lib -lsuperlu_mt_openmp
	#CINCS   +=-I/home/srodrigu/Software/SuperLU_MT_mkl/SRC


	DEFINES +=-D_SUPERLU_BACKEND_

	# CLIBS   +=-L/home/srodrigu/Software/SuperLU_MT/lib -lsuperlu_mt_openmp
	# CINCS   +=-I/home/srodrigu/Software/SuperLU_MT/SRC

	CLIBS   +=-L/home/srodrigu/Software/SuperLU_MT_3.1/lib -lsuperlu_mt_OPENMP
	CINCS   +=-I/home/srodrigu/Software/SuperLU_MT_3.1/SRC

	
	OBJS    += spike_superlu.o

else ifeq ($(BACKEND),PARDISO)
	DEFINES += -D_PARDISO_BACKEND_
	OBJS    += spike_pardiso.o

else
	NVCC     = nvcc
	NVCFLAGS =-dlink -ccbin=icc #-Xcompiler -fopenmp -O2
	NVCLIBS  +=-L/usr/local/cuda/lib64 -lcusolver -lcusparse
	NVCINCS  +=-I/usr/local/cuda/include

	CINCS   +=-I/usr/local/cuda/include
	CLIBS   +=-L/usr/local/cuda/lib64 -lcudart -lcusolver -lcusparse

	DEFINES += -D_CUDA_ENABLED_ -D_NVIDIA_BACKEND_

	OBJS    += spike_cuda.o
endif


ifeq ($(DEBUG),YES)
	DEFINES += -D_ENABLE_TESTING_
endif



TARTGET=main


LIBSPIKE = libspike.a

all:main tests

#main: main.o $(LIBSPIKE)
#	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS) -L. -lspike

main: main.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

main.o:main.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

#libspike.a:$(OBJS)
#	ar rc $@ $+
#	ranlib libspike.a

spike_interfaces.o:spike_interfaces.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_datatypes.o:spike_datatypes.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_memory.o:spike_memory.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_matrix.o:spike_matrix.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_algebra.o:spike_algebra.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_pardiso.o:spike_pardiso.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_superlu.o:spike_superlu.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_cuda.o:spike_cuda.c
	$(NVCC) $(NVCFLAGS) $(DEFINES) $(NVCLIBS) $+ -c

spike_analysis.o:spike_analysis.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_common.o:spike_common.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c



# Compile different tests
tests:superlu_test

# inner solver solution test for small matrix
innersolver: innersolvercheck.o $(OBJS) $(LIBSPIKE)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS) -L. -lspike

innersolvercheck.o: innersolvercheck.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c


# inner solver solution test for small matrix
superlu_test: superlu_test.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

superlu_test.o: superlu_test.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c





# # split test
# split: split.o $(OBJS)
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

# split.o: split.c
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# # bandwidth test
# bandwidth: bandwidth.o $(OBJS)
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

# bandwidth.o: bandwidth.c
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# # bandwidth test
# reduced: reduced.o $(OBJS)
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

# reduced.o: reduced.c
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# # new reduced system assembly functions
# mpireduced: mpireduced.o $(OBJS)
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

# mpireduced.o: mpireduced.c
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# # inner solver solution test for small matrix
# innersolver: innersolvercheck.o $(OBJS)
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

# innersolvercheck.o: innersolvercheck.c
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# # inner solver solution test for small matrix
# sequential: spike_sequential.o $(OBJS)
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

# spike_sequential.o: spike_sequential.c
# 	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

.PHONY:all run clean

all:$(TARGET)

run:
	./main

clean:
	rm -r main libspike.a *.o
