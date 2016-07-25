COMPILER=INTEL

ifeq ($(COMPILER),INTEL)
	CC      =mpiicc
	CFLAGS  =-qopenmp -std=c99 -g -restrict -O2
	#CLIBS_MAC =-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
	CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
	CINCS   =-I../OpenMP -I${MKLROOT}/include
else
	CC      =mpicc
	CFLAGS  =-std=c99 -g -O2
	CLIBS   =
	CINCS   =-I../OpenMP
endif

DEFINES =-D_DATATYPE_D_ -D_ENABLE_TESTING_ -D_MPI_SUPPORT_ -D_DEBUG # -D_DEBUG_MATRIX_

TARTGET=main
OBJS = spike_datatypes.o spike_memory.o spike_matrix.o \
			 spike_algebra.o spike_analysis.o spike_common.o spike_mpi.o

main: main.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

main.o:main.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_datatypes.o:../OpenMP/spike_datatypes.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_memory.o:../OpenMP/spike_memory.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_matrix.o:../OpenMP/spike_matrix.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_algebra.o:../OpenMP/spike_algebra.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_analysis.o:../OpenMP/spike_analysis.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_common.o:../OpenMP/spike_common.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_mpi.o:spike_mpi.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c


# split test
split: mpi_split.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

mpi_split.o: mpi_split.c
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
