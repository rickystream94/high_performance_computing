TARGET	= poisson
OBJS	= main.o matrixlib.o jacobi.o gauss_seidel.o

OPT	= -g -fast -lm -xopenmp -xvpara
CCC	= CC
CFLAGS	= $(OPT) -Wall
DEPS = jacobi.h matrixlib.h gauss_seidel.h

$(TARGET): $(OBJS) 
	$(CCC) $(CFLAGS) -o $@ $^

clean:
	@/bin/rm -f *.o core*

realclean: clean
	@rm -f $(TARGET)

# dependencies
%.o: %.c $(DEPS)
	$(CCC) -c $(CFLAGS) $< -o $@