#!/bin/bash
#BSUB -J 07timing
#BSUB -q hpc
#BSUB -W 3
#BSUB -R rusage[mem=16GB]
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -o hpc_files/07timing%J.out
#BSUB -e hpc_files/07timing%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

time python Ex07numbaJacobi.py 10