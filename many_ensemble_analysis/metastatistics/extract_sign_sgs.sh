#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --job-name=sign_sgs
#SBATCH -M XXXXX
#SBATCH --partition=XXXXX
#SBATCH --array=0-100

srun python3 extract_sign_sgs.py
