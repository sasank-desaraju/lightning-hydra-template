#!/bin/bash
#SBATCH --job-name=Liver-Example
#SBATCH --mail-user=sasank.desaraju@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output ./slurm/logs/my_job-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"


# Load compiler
# module load gcc
# echo "gcc version is $(gcc --version)"

# Load env
module load conda
conda activate cv

# HPG actually recommends prepending the path to the environment instead of loading it.
export PATH=/home/sasank.desaraju.conda/envs/cv/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sasank.desaraju.conda/envs/cv/bin

echo "PATH is "
echo PATH $PATH /$PATH

# Python script
python /blue/prismap-ai-core/sasank.desaraju/projects/lightning-hydra-template/src/train.py experiment=spleen-example
