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
io.o: io.c io.h
	$(COMPILE) io.c
data.o: data.c data.h
	$(COMPILE) data.c
main.o: main.c data.h io.h
	$(COMPILE) main.c
