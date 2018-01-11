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

SIZES=["10 15 20 25 30 40 50 60 80 100"]
K="1000000"
d="0.000001"
TYPE="jacobi"

for n in $SIZES
do
    ./$EXECUTABLE $TYPE $n $K $d
done

### ./$EXECUTABLE $(cat input.in) ### Use this version if you want to read arguments from command line