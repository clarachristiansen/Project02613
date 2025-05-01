#!/bin/bash
#BSUB -J 09cupyprofiler
#BSUB -q gpuv100
#BSUB -u s214659@dtu.dk
#BSUB -W 5
#BSUB -R rusage[mem=8GB]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu32gb]"
#BSUB -o hpc_files/09cupyprofiler%J.out
#BSUB -e hpc_files/09cupyprofiler%J.err

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

nsys profile -o 09cupy02 python 09cupy.py 10