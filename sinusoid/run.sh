#!/bin/bash
#SBATCH --job-name=sinusoid
#SBATCH --output=/home-mscluster/djarvis/masters/sinusoid/result.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch
export CUDA_DIR=/usr/local/cuda-10.0-alternative
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.0-alternative
python /home-mscluster/djarvis/masters/sinusoid/sinu.py
