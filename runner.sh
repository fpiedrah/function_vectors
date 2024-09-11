#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=fpiedrah@brown.edu

#SBATCH -t 5:00:00
#SBATCH -p gpu-he --gres=gpu:1

module load cuda
python -m function_vectors.main
