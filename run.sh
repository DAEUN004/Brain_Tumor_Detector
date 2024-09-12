#!/bin/bash
#SBATCH --partition=M2
#SBATCH --qos=q_a_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --job-name=MyJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err 

module load anaconda
eval "$(conda shell.bash hook)"
conda activate brain_tumor_detector

python main.py