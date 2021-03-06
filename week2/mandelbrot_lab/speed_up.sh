#BSUB -J speedup
#BSUB -o speedup_%J.out
#BSUB -e speedup_%J.err
#BSUB -q hpcintro
#BSUB -n 12
#BSUB -W 15

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

module load studio

# define the executable here
#
EXECUTABLE=mandelbrot

# keep idle threads spinning (needed to monitor idle times!)
export OMP_WAIT_POLICY=active

# set max number of threads to loop for
# MAX_THREADS=$LSB_DJOB_NUMPROC
SIZES="128 256 512 1024 2048 4096"
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
# for i in `seq 1 $MAX_THREADS`
for i in $SIZES
do
    # set number of threads for current execution
    # export OMP_NUM_THREADS=$i
    ./$EXECUTABLE $i
done