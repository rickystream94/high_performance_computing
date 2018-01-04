#!/bin/bash
### General Options
### -- specify job name --
#BSUB -J tune_lab

### -- Ask for number of cores (default: 1) -- 
#BSUB -n 4

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o tune_lab_%J.out
#BSUB -e tune_lab_%J.err

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

NPARTS="1000 2000 3000 4000 5000 7500 10000 15000 20000 25000"
LOOPS=10000
LOGEXT=dat
lscpu
/bin/rm -f external.$LOGEXT internal.$LOGEXT
for particles in $NPARTS
do
    ./external $LOOPS $particles | grep -v CPU >> external.$LOGEXT
    ./internal $LOOPS $particles | grep -v CPU >> internal.$LOGEXT
done

# time to say 'Good bye' ;-)
#
exit 0
