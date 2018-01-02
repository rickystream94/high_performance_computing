CC = gcc
OBJECTS = main.o data.o io.o
OPT = -g -O3
WARN = -Wall
CFLAGS = $(OPT) $(WARN)
COMPILE = $(CC) -c $(CFLAGS)

excer2: $(OBJECTS)
	$(CC) $(OBJECTS) $(CFLAGS) -o excer2
	echo "Done."
clean:
	rm -f *.o
io.o: io.h
	$(COMPILE) io.c
data.o: data.h
	$(COMPILE) data.c
main.o: io.h data.h
	$(COMPILE) main.c
