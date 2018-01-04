CC = gcc
DEBUG = -g
OPT =  -O3
ATLAS = -L /usr/lib64/atlas -lsatlas
CFLAGS = $(DEBUG) $(OPT)
COMPILE = $(CC) $(CFLAGS) $(ATLAS)
OBJECTS = check.o

project: $(OBJECTS)
	$(COMPILE) $(OBJECTS) $(CFLAGS) -o project

clean:
	rm -f *.o

check.o: check.c matmult.h

