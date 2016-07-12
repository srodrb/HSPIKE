CC      =icc
CFLAGS  =-qopenmp -std=c99 -g -restrict -O2
#CLIBS_MAC =-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
CINCS   =-I${MKLROOT}/include
DEFINES =-D_DATATYPE_D_ -D_ENABLE_TESTING_ # -D_DEBUG_MATRIX_

TARTGET=main
OBJS = spike_datatypes.o spike_memory.o spike_matrix.o \
			 spike_algebra.o spike_analysis.o spike_common.o

main: main.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

main.o:main.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_datatypes.o:spike_datatypes.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_memory.o:spike_memory.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_matrix.o:spike_matrix.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_algebra.o:spike_algebra.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_analysis.o:spike_analysis.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_common.o:spike_common.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c



# split test
split: split.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

split.o: split.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# bandwidth test
bandwidth: bandwidth.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

bandwidth.o: bandwidth.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# bandwidth test
reduced: reduced.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

reduced.o: reduced.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

# new reduced system assembly functions
mpireduced: mpireduced.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

mpireduced.o: mpireduced.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c


.PHONY:all run clean

all:$(TARGET)

run:
	./main

clean:
	rm -r $(TARGET) *.o
