CC      =icc
CFLAGS  =-std=c99 -O2
CINCS   = 
CLIBS   =
DEFINES =-DSPRECISION -D_ENABLE_TESTING_

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



# Build tests
split: split.o $(OBJS)
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -o $@ $(CLIBS)

split.o: split.c
	$(CC) $(CFLAGS) $(DEFINES) $(CINCS) $+ -c



.PHONY:all run clean

all:$(TARGET)

run:
	./main

clean:
	rm -r $(TARGET) *.o
