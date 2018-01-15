### Bash Script used to produce needed output to plot speed up, with increasing number of threads
### Please set a fixed SIZE!!!

#BSUB -J speedup
#BSUB -o speedup_%J.out
#BSUB -e speedup_%J.err
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

# keep idle threads spinning (needed to monitor idle times!)
export OMP_WAIT_POLICY=active

# set max number of threads to loop for
MAX_THREADS=$LSB_DJOB_NUMPROC

# arguments
SIZE="4000" ### Use this value for the comparison with Mandelbrot
###SIZE="2000"
TYPE="jacobi"
K="100" # keep it low to avoid long execution times
D="0.000001"

printf "N=$SIZE\n"
for i in `seq 1 $MAX_THREADS`
do
    # set number of threads for current execution
    export OMP_NUM_THREADS=$i
    ./$EXECUTABLE $TYPE $SIZE $K $D
done