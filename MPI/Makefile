COMPILER =intel
BACKEND  =pardiso
DEBUG    =no

# Data type definition
DEFINES +=-D_MPI_SUPPORT_ -D_DATATYPE_D_ -D_NDEBUG

OBJS = spike_datatypes.o spike_matrix.o \
       spike_algebra.o spike_analysis_dm.o spike_common.o spike_mpi.o \
	   spike_dm.o spike_blocking.o 

ifeq ($(COMPILER),intel)
	CC      =mpiicc
	CFLAGS  =-qopenmp -std=c99 -O2 -g -restrict

	# multi-threaded version
	CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
	CINCS   =-I${MKLROOT}/include
	
	# sequential version
	# CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
	# CINCS   =-I${MKLROOT}/include
	
	CINCS   +=-I../OpenMP
else
	CC      =mpicc
	CFLAGS  =-std=c99 -g -O2
	CLIBS   =
	CINCS   +=-I../OpenMP
endif

ifeq ($(BACKEND),superlu)
	#CLIBS   +=-L/home/srodrigu/Software/SuperLU_MT_mkl/lib -lsuperlu_mt_openmp
	#CINCS   +=-I/home/srodrigu/Software/SuperLU_MT_mkl/SRC

	DEFINES +=-D_SUPERLU_BACKEND_

	# CLIBS   +=-L/home/srodrigu/Software/SuperLU_MT/lib -lsuperlu_mt_openmp
	# CINCS   +=-I/home/srodrigu/Software/SuperLU_MT/SRC

	CLIBS   +=-L/home/srodrigu/Software/SuperLU_MT_3.1/lib -lsuperlu_mt_OPENMP
	CINCS   +=-I/home/srodrigu/Software/SuperLU_MT_3.1/SRC

	
	OBJS    += spike_superlu.o
endif

ifeq ($(BACKED),pardiso)
	DEFINES +=-D_PARDISO_BACKEND_
	OBJS    += spike_pardiso.o spike_memory_host.o
endif

ifeq ($(BACKEND),cusolver)
	NVCC      = nvcc
	#–default-stream per-thread
	NVCFLAGS  =-O3 --gpu-architecture=compute_37 -G -dlink -ccbin=icc -Xcompiler # -std=c99 #-Xcompiler -fopenmp -O2
	NVCLIBS  +=-L/usr/local/cuda/lib64 -lcusolver -lcusparse
	NVCINCS  +=-I/usr/local/cuda/include

	CINCS    +=-I/usr/local/cuda/include
	CLIBS    +=-L/usr/local/cuda/lib64 -lcudart -lcusolver -lcusparse

	DEFINES  += -D_CUDA_ENABLED_ -D_NVIDIA_BACKEND_

	OBJS     += spike_cuda.o spike_memory_cuda.o
endif

ifeq  ($(DEBUG),YES)
	DEFINES += -D_ENABLE_TESTING_ -D_DEBUG
endif

DEFINES +=-D_PARDISO_BACKEND_
OBJS    += spike_pardiso.o spike_memory_host.o

TARTGET=main

main: main.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

main.o:main.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_datatypes.o:../OpenMP/spike_datatypes.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_pardiso.o:../OpenMP/spike_pardiso.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_superlu.o:../OpenMP/spike_superlu.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_memory_host.o:../OpenMP/spike_memory_host.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_matrix.o:../OpenMP/spike_matrix.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_algebra.o:../OpenMP/spike_algebra.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_analysis.o:../OpenMP/spike_analysis.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_analysis_dm.o:spike_analysis_dm.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_common.o:../OpenMP/spike_common.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_mpi.o:spike_mpi.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_dm.o:spike_dm.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_blocking.o:spike_blocking.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_cuda.o:spike_cuda.c
	$(NVCC) $(NVCFLAGS) $(DEFINES) $(NVCLIBS) $+ -c

spike_memory_cuda.o:spike_memory_cuda.c
	$(NVCC) $(NVCFLAGS) $(DEFINES) $(NVCLIBS) $+ -c


# split test
split: mpi_split.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

mpi_split.o: mpi_split.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# bandwidth test
generalTest: generalTest.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

generalTest.o: generalTest.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# LU SIZE TEST
testSizeLU: testSizeLU.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

testSizeLU.o: testSizeLU.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# bandwidth test
reduced: reduced.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

reduced.o: reduced.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# macros test
macros: mpi_macros.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

mpi_macros.o: mpi_macros.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

.PHONY:all run clean

all:$(TARGET)

run:
	./main

clean:
	rm -r $(TARGET) *.o
