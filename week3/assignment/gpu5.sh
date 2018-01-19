#BSUB -J gpu5
#BSUB -o gpu5_%J.out
#BSUB -e gpu5_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 4
#BSUB -W 30

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify memory usage --
#BSUB -R "rusage[mem=1024MB]"

# define the executable here
#
module load cuda/9.1
EXECUTABLE=matmult_f.nvcc

SIZES="32 48 96 112 144 160 240 560 800 1120 1280 1440 1600 1920 2400 3200 4000 5600 6400 7200 8000 8800 9600 9920"
MAX_IT=100
COMPARE=0

for i in $SIZES
do
    MATMULT_COMPARE=$COMPARE ./$EXECUTABLE gpu5 $i $i $i
done