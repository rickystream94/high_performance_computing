#BSUB -J gpu3_right_neighbor
#BSUB -o gpu3_right_neighbor_%J.out
#BSUB -e gpu3_right_neighbor_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 20
#BSUB -W 30

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

# define the executable here
#
module load cuda/9.1
EXECUTABLE=matmult_f.nvcc

SIZES="50 100 200 300 500 750 1000 1250 1500 1750"
MAX_IT=1

# First round: lib
for i in $SIZES
do
    MFLOPS_MAX_IT=$MAX_IT ./$EXECUTABLE gpu3 $i $i $i
done