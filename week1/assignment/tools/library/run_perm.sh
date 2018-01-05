### General Options
### -- specify job name --
#BSUB -J permutations

### -- Ask for number of cores (default: 1) -- 
#BSUB -n 4

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o permutations_%J.out
#BSUB -e permutations_%J.err

### -- Specify the queue name --
#BSUB -q hpcintro

### -- Specify walltime limit -- 
#BSUB -W 10

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

### -- Specify model and feature(s) --
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "select[avx2]"

### -- Notify by email when job starts --
#BSUB -B

### -- Notify by email when job ends --
#BSUB -N

SIZES = "10 20 30 40 50 60 70 80 90 100"

for size in $SIZES
do
    matmult_c.gcc mkn $size $size $size
    matmult_c.gcc mnk $size $size $size
    matmult_c.gcc kmn $size $size $size
    matmult_c.gcc knm $size $size $size
    matmult_c.gcc nkm $size $size $size
    matmult_c.gcc nmk $size $size $size
done 