#!/bin/bash
#BSUB -J 08_5cudaBig
#BSUB -q gpuv100
#BSUB -W 02:00
#BSUB -R rusage[mem=8GB]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -u s214656@dtu.dk
#BSUB -o hpc_files/08_5cudaBig%J.out
#BSUB -e hpc_files/08_5cudaBig%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

time python 08_5cuda.py 4571
