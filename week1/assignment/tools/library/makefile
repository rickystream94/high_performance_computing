TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= matmult.o

CC = gcc
OPT =  -g -O3
PIC	= -fPIC
ATLAS = -L /usr/lib64/atlas -lsatlas
CFLAGS = $(OPT) $(PIC) $(XOPTS)

SOFLAGS = -shared 
XLIBS = 

COMPILE = $(CC) -o $@  $(LIBOBJS) $(SOFLAGS) $(CFLAGS) $(ATLAS) $(XLIBS)

$(TARGET): $(LIBOBJS)
	$(COMPILE)

clean:
	rm -f *.o

matmult.o: matmult.c
	$(CC) -c $(CFLAGS) $< -o $@

