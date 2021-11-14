#!/bin/bash
#SBATCH --job-name=mnist_class
#SBATCH --output=/home-mscluster/djarvis/masters/mnist/result_mnist.txt
#SBATCH --ntasks=1
#SBATCH --partition=ha
export CUDA_DIR=/usr/local/cuda-10.0-alternative
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.0-alternative
echo $CUDA_DIR
echo $XLA_FLAGS
python /home-mscluster/djarvis/masters/mnist/mnist.py
