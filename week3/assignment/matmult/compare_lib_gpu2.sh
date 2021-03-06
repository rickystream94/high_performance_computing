#BSUB -J compare_lib_gpu2
#BSUB -o compare_lib_gpu2_%J.out
#BSUB -e compare_lib_gpu2_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 20
#BSUB -W 10

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=1024MB]"

# define the executable here
#
module load cuda/9.1
EXECUTABLE=matmult_f.nvcc
### export MFLOPS_MAX_IT=1

SIZES="50 100 200 300 500 750 1000 1250 1500 1750 2000 2500 3000 3500 4000 4500 5000 5500 6000 7500 8000 8500 9000 10000"
COMPARE=0
MAX_IT=100

# First round: lib
for i in $SIZES
do
    MATMULT_COMPARE=$COMPARE ./$EXECUTABLE lib $i $i $i
done

# Second round: gpu2
for i in $SIZES
do
    MATMULT_COMPARE=$COMPARE ./$EXECUTABLE gpu2 $i $i $i
done