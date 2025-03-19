#!/bin/bash
#BSUB -J 02timing
#BSUB -q hpc
#BSUB -u s214659@dtu.dk
#BSUB -W 15
#BSUB -R rusage[mem=16GB]
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -o hpc_files/02timing%J.out
#BSUB -e hpc_files/02timing%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

time python simulate.py 10