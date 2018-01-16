#BSUB -J speedup
#BSUB -o speedup_%J.out
#BSUB -e speedup_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 4
#BSUB -W 15

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

# define the executable here
#
EXECUTABLE=mandelbrot

# set max number of threads to loop for
#THREADS_PER_BLOCK="2 4 8 16 32"
SIZES="128 256 512 1024 2048 4096"
for i in $SIZES
do
    ./$EXECUTABLE $i
done