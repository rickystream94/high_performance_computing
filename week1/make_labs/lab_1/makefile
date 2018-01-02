CC = gcc
OBJECTS = main.o data.o io.o

excer1: $(OBJECTS)
	$(CC) $(OBJECTS) -o excer1
	echo "Done."
clean:
	rm -f *.o
data.o: data.c data.h
	$(CC) -c data.c
io.o: io.c io.h
	$(CC) -c io.c
main.o: main.c data.h io.h
	$(CC) -c main.c
