#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=sign_ses
#SBATCH -M XXXXX
#SBATCH --partition=XXXXX
#SBATCH --array=0-100

srun python3 extract_sign_ses.py
