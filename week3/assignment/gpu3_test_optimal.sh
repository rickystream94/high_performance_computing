#BSUB -J gpu3_test_optimal
#BSUB -o gpu3_test_optimal_%J.out
#BSUB -e gpu3_test_optimal_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 20
#BSUB -W 30

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=1024MB]"

# define the executable here
#
module load cuda/9.1
EXECUTABLE=matmult_f.nvcc

SIZES="50 100 200 300 500 750 1000 1250 1500 1750 2000 2500 3000 3500 4000 4500 5000 5500 6000 7500 8000 8500 9000 10000"
MAX_IT=100
COMPARE=0

# First round: lib
for i in $SIZES
do
    MATMULT_COMPARE=$COMPARE ./$EXECUTABLE gpu3 $i $i $i
done