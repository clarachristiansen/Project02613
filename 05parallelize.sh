#!/bin/bash
#BSUB -J 05parallelize
#BSUB -q hpc
#BSUB -u s214659@dtu.dk
#BSUB -W 10:00
#BSUB -R rusage[mem=16GB]
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6226R]" 
#BSUB -o hpc_files/05parallelize%J.out
#BSUB -e hpc_files/05parallelize%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

for i in 1 2 4 8 16;
do
    time python 05parallelize.py 50 $i
done