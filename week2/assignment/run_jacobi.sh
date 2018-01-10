#BSUB -J jacobi
#BSUB -o jacobi_%J.out
#BSUB -e jacobi_%J.err
#BSUB -q hpcintro
#BSUB -n 20
#BSUB -W 15

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

module load studio

# define the executable here
#
EXECUTABLE=poisson

./$EXECUTABLE $(cat input.in)