#!/bin/bash
### General Options
### -- specify job name --
#BSUB -J lib_perf

### -- Ask for number of cores (default: 1) -- 
#BSUB -n 4

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o lib_perf_%J.out
#BSUB -e lib_perf_%J.err

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

SIZES="5 10 15 20 30 40 50 60 80 90 100 120 140 170 200 250 300 400 500 700 900 1000 1300 1500"

for size in $SIZES
do
    matmult_c.gcc lib $size $size $size
done 