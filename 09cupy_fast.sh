#!/bin/bash
#BSUB -J 09cupy_fast
#BSUB -q gpuv100
#BSUB -u s214659@dtu.dk
#BSUB -W 5
#BSUB -R rusage[mem=8GB]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -o hpc_files/09cupy_fast%J.out
#BSUB -e hpc_files/09cupy_fast%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

time python 09cupy_fast.py 10