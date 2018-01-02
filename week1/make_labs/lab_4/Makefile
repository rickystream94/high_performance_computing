FC = gfortran
F90C	= f90
F90FLAGS= -g -O3
OBJECTS	= f90main.o
### .SUFFIXES = .f90

f90project: $(OBJECTS)
	$(FC) $(F90FLAGS) -o $@ $(OBJECTS)

clean:
	@/bin/rm -f *.o core

f90main.o:
	$(FC) -c f90main.f90
