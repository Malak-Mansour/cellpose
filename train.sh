#!/bin/bash
#SBATCH --job-name=cellpose_debug
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=124G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-49,gpu-05
#SBATCH --no-requeue

# Activate your conda environment
source ~/.bashrc
conda activate cellpose

# Run the Python script
python train.py
