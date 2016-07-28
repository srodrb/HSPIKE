COMPILER=INTEL

ifeq ($(COMPILER),INTEL)
	CC      =icc
	CFLAGS  =-qopenmp -std=c99 -g -O2 -restrict
	#CLIBS_MAC =-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
	CLIBS   =-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl
	CINCS   =-I${MKLROOT}/include -I.

	#CLIBS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl
	#CINCS = -qopenmp -I${MKLROOT}/include
else
	CC      =gcc
	CFLAGS  =-std=c99 -g -O2
	CLIBS   =
	CINCS   =
endif

DEFINES =-D_DATATYPE_D_ -D_ENABLE_TESTING_
#DEFINES =-D_DATATYPE_Z_ -D_ENABLE_TESTING_

TARTGET=main

OBJS = spike_datatypes.o spike_memory.o spike_matrix.o \
			 spike_algebra.o spike_analysis.o spike_common.o \
			 spike_interfaces.o

SPIKELIB = libspike.a

main: main.o $(SPIKELIB)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS) -L. -lspike

main.o:main.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

libspike.a:$(OBJS)
	ar rc $@ $+
	ranlib libspike.a

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

spike_analysis.o:spike_analysis.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c

spike_common.o:spike_common.c
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
