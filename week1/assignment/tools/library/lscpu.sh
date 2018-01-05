### General Options
### -- specify job name --
#BSUB -J lscpu

### -- Ask for number of cores (default: 1) -- 
#BSUB -n 4

### -- Ask for number of hosts (nodes) --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o lscpu_%J.out
#BSUB -e lscpu_%J.err

### -- Specify the queue name --
#BSUB -q hpcintro

### -- Specify walltime limit -- 
#BSUB -W 2

### -- Specify memory usage --
#BSUB -R "rusage[mem=512MB]"

### -- Specify model and feature(s) --
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "select[avx2]"

### -- Notify by email when job starts --
#BSUB -B

### -- Notify by email when job ends --
#BSUB -N

lscpu