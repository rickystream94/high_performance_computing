#!/bin/bash
### General Options
### -- specify job name --
#BSUB -J blk_perf

### -- Ask for number of cores (default: 1) -- 
#BSUB -n 4

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o blk_perf_%J.out
#BSUB -e blk_perf_%J.err

### -- Specify the queue name --
#BSUB -q hpcintro

### -- Specify walltime limit -- 
#BSUB -W 15

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

### -- Specify model and feature(s) --
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "select[avx2]"

### -- Notify by email when job starts --
#BSUB -B

### -- Notify by email when job ends --
#BSUB -N

###SIZE="60" #Size for L1 cache
###SIZE="140" #Size for L2 cache
SIZE="1500" #Size for L3 cache
###BLOCK_SIZES="1 3 5 8 10 15 20 23 25 28 30 33 35 38 40 45 50 53 55 58" #Block Sizes for L1 cache
###BLOCK_SIZES="1 3 5 8 10 15 20 30 50 70 80 90 100 105 110 115 120 125 128 135" #Block Sizes for L2 cache
BLOCK_SIZES="1 5 10 20 50 100 200 300 500 700 800 900 1000 1100 1150 1180 1200 1300 1400 1450" #Block Sizes for L3 cache
for bs in $BLOCK_SIZES
do
    matmult_c.gcc blk $SIZE $SIZE $SIZE $bs
done